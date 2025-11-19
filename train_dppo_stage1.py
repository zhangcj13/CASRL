import os

os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase_PPOS_STAGE1'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import sys
import gym

gym.logger.set_level(40)
from gym_collision_avoidance.envs import Config
# Config.set_display() # Visual training
from ga3c.Environment import Environment

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from dppo.model import Model
from dppo.train import ReplayMemory

from torch.utils.tensorboard import SummaryWriter
import time

class Params():
    def __init__(self):
        # self.batch_size = 1000
        self.lr = 2e-4 # 4agents:  2e-4, 10agents: 2e-5
        self.epoch_lr = {0    :2e-4,
                        #  1    :3e-4,
                         5000 :1e-4,
                         10000:5e-5,
                         20000:2e-5,}
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 32  # 1000
        # self.exploration_size = 1024  # 1000
        self.num_processes = 16 # 4
        self.update_treshold = self.num_processes - 1
        self.max_episode_length = 25000
        self.max_grad_norm = 0.5
        self.seed = 1
        self.env_name = '"CollisionAvoidance-v0"'
        self.save_freq=1000

        self.obs_size = Config.HOST_AGENT_OBSERVATION_LENGTH
        self.oas_size = Config.OTHER_AGENT_OBSERVATION_LENGTH
        self.action_dim = 11

        self.ampl2goal = {0  : 0.20,
                          5  : 0.20,
                          200: 0.08,}
        self.max_num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT

        self.coeff_entropy = 0.02

        self.exploration_size = self.num_processes*self.num_steps*2
        self.batch_size = self.exploration_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pre_load_epoch=0 # 4agents:  0, 10agents: 25000

def ppo_update(params, memory, policy, optimizer):
    # memory.norm_data(6)
    advs_mean, advs_std = memory.get_mean_std(6)

    sum_loss=0
    for k in range(params.num_epoch):
        # new mini_batch
        sampled_states, sampled_actions, sampled_actionprobs, sampled_returns, sampled_values, sampled_rewards, sampled_advantages = memory.sample(
            params.batch_size)

        sampled_advantages = (sampled_advantages - advs_mean) / advs_std

        sampled_actions = Variable(torch.from_numpy(sampled_actions)).float().to(params.device)
        sampled_actionprobs = Variable(torch.from_numpy(sampled_actionprobs)).float().to(params.device)
        sampled_returns = Variable(torch.from_numpy(sampled_returns)).float().to(params.device)
        sampled_advantages = Variable(torch.from_numpy(sampled_advantages)).float().to(params.device)

        new_value, new_actionprob, dist_entropy = policy.evaluate_actions(sampled_states, sampled_actions)

        sampled_actionprobs = sampled_actionprobs.view(-1, 1)
        # ratio = torch.exp(torch.log(new_actionprob) - torch.log(sampled_actionprobs))
        ratio = torch.exp(new_actionprob - sampled_actionprobs)

        sampled_advs = sampled_advantages.view(-1, 1)
        surrogate1 = ratio * sampled_advs
        surrogate2 = torch.clamp(ratio, 1 - params.clip, 1 + params.clip) * sampled_advs
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        sampled_targets = sampled_returns.view(-1, 1)

        value_loss = F.mse_loss(new_value, sampled_targets)

        # actor_loss = policy_loss - params.coeff_entropy * dist_entropy  # shape(mini_batch_size X 1)
        # optimizer['actor'].zero_grad()
        # actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 0.5)
        # optimizer['actor'].step()
        # optimizer['critic'].zero_grad()
        # value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), 0.5)
        # optimizer['critic'].step()

        loss = policy_loss + value_loss - params.coeff_entropy * dist_entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(policy.parameters(), params.max_grad_norm)
        optimizer.step()

        sum_loss+=loss

    avg_loss=sum_loss.detach().cpu().numpy()/params.num_epoch

    return avg_loss


def run(params, policy, optimizer, envs, policy_path,tb_writer):

    memory = ReplayMemory(params.exploration_size)

    episode_length = params.pre_load_epoch
    global_update = params.pre_load_epoch

    game_overs = np.full((params.num_processes), True, dtype=bool)
    # reset all training envs

    for i in range(params.num_processes):
        envs[i].reset()

        # define all data buf for training
    states = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}
    actions = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}
    rewards = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}
    values = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}
    returns = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}
    advantages = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}
    actionprobs = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}
    dones = {n: {i: [] for i in range(params.max_num_agents)} for n in range(params.num_processes)}

    step_counts = np.zeros((params.num_processes, params.max_num_agents))
    # cur_which_agents_done = np.full((params.num_processes, params.max_num_agents), False, dtype=bool)
    which_agents_dones = {n: np.full((params.max_num_agents), False, dtype=bool) for n in range(params.num_processes)}
    which_agents_done_and_traineds = {n: np.full((params.max_num_agents), False, dtype=bool) for n in
                                      range(params.num_processes)}

    cum_reward_sum_loggers = {n: np.zeros((params.max_num_agents)) for n in range(params.num_processes)}
    ampl2goal = params.ampl2goal[0] # 4agents:  params.ampl2goal[0], 10agents: 0.08

    # time_count = 0
    cost_time = 1.0

    while episode_length < params.max_episode_length:
        start_time = time.time()
        episode_length += 1

        if episode_length in params.epoch_lr.keys():
            for p in optimizer.param_groups:
                p['lr']=params.epoch_lr[episode_length]
                print('epoch :{}, lr={}'.format(episode_length,optimizer.state_dict()['param_groups'][0]['lr']))
        if episode_length in params.ampl2goal.keys():
            ampl2goal = params.ampl2goal[episode_length]
            print('============= epoch :{}, ampl2goal={} ============='.format(episode_length, ampl2goal))

        av_reward = 0
        cum_done = 0
        t = -1
        finish_status = {'Reach Goal': 0,
                         'Crashed': 0,
                         'Time out': 0, }
        while len(memory) < params.exploration_size:
            t += 1
            # Perform every agent's step<num_steps and no agent is done
            state_idxs = []
            states_list = []

            for ip in envs.keys():
                # cal the actions
                # env_actions = {}
                for i, agent_observation in enumerate(envs[ip].latest_observations):
                    is_agent_running = agent_observation[0]
                    if not is_agent_running or which_agents_dones[ip][i]:
                        continue
                    state = np.asarray([agent_observation])

                    states_list.append(state)

                    state_idxs.append([ip,i])

                    # v, a, a_prob, action_id = policy.generate_action(state)
                    # # print(action_id)
                    # # env_actions[i] = action_id
                    # states[ip][i].append(state)
                    # actions[ip][i].append(a)
                    # values[ip][i].append(v)
                    # actionprobs[ip][i].append(a_prob)

            states_list = np.asarray(states_list).squeeze(1)
            v_list, a_list, a_prob_list, action_id_list = policy.generate_action(states_list,multi_actions_flag=True)

            multi_env_actions = {}
            for j, idb in enumerate(state_idxs):
                ip, i = idb
                if ip not in multi_env_actions.keys():
                    multi_env_actions[ip] = {}
                multi_env_actions[ip][i] = action_id_list[j]
                states[ip][i].append(states_list[[j]])
                actions[ip][i].append(a_list[[j]])
                values[ip][i].append(v_list[[j]])
                actionprobs[ip][i].append(a_prob_list[[j]])

            for ip in envs.keys():

                env_actions = multi_env_actions[ip]

                cur_rewards, game_over, infos = envs[ip].step([env_actions], rank, t, ampl2goal=ampl2goal)
                cur_rewards = cur_rewards[0]

                which_agents_done = infos[0]['which_agents_done']
                which_agents_learning = infos[0]['which_agents_learning']
                num_agents_running_ga3c = np.sum(list(which_agents_learning.values()))
                which_agents_status_dict = infos[0]['which_agents_status_dict']

                which_agents_dones[ip] = which_agents_done
                game_overs[ip] = game_over

                for i in which_agents_learning.keys():
                    # Loop through all feedback from environment (which may not be equal to Config.MAX_NUM_AGENTS)
                    if not which_agents_learning[i] or which_agents_done_and_traineds[ip][i]:
                        continue
                    reward = cur_rewards[i]
                    done = which_agents_done[i]
                    # cur_which_agents_done[ip][i] = done
                    dones[ip][i].append(done)
                    cum_reward_sum_loggers[ip][i] += cur_rewards[i]
                    rewards[ip][i].append(np.array([[reward]]))

                    step_counts[ip][i] += 1

                    if which_agents_done[i] or step_counts[ip][i]>= params.num_steps:
                        # one last step
                        R = np.zeros((1, 1))
                        if which_agents_done[i]:
                            which_agents_done_and_traineds[ip][i] = True
                        else:
                            agent_observation = envs[ip].latest_observations[i]
                            v, _, _, _ = policy.generate_action(np.asarray([agent_observation]))
                            R = v
                        values[ip][i].append(R)
                        # compute returns and GAE(lambda) advantages:
                        num_step = len(rewards[ip][i])
                        gae = np.zeros((1, 1))
                        for t in range(num_step - 1, -1, -1):
                            delta = rewards[ip][i][t] + params.gamma * values[ip][i][t + 1] - values[ip][i][t]
                            gae = delta + params.gamma * params.gae_param * gae
                            advantages[ip][i].insert(0, gae)
                            R = gae + values[ip][i][t]
                            returns[ip][i].insert(0, R)

                        memory.push([states[ip][i], actions[ip][i], actionprobs[ip][i], returns[ip][i], values[ip][i],
                                    rewards[ip][i], advantages[ip][i]])
                        
                        states[ip][i] = []
                        actions[ip][i] = []
                        actionprobs[ip][i] = []
                        returns[ip][i] = []
                        rewards[ip][i] = []
                        values[ip][i] = []
                        advantages[ip][i] = []
                        step_counts[ip][i] = 0

                if game_over:
                    cum_done += 1
                    av_reward += np.sum(cum_reward_sum_loggers[ip]) / num_agents_running_ga3c
                    cum_reward_sum_loggers[ip] = np.zeros((params.max_num_agents))

                    for i in which_agents_learning.keys():
                        if not which_agents_learning[i]:
                            continue
                        finish_status[which_agents_status_dict[i]] += 1

                    envs[ip].reset()
                    # cur_which_agents_done[ip, :] = False
                    which_agents_dones[ip] = np.full((params.max_num_agents), False, dtype=bool)
                    which_agents_done_and_traineds[ip] = np.full((params.max_num_agents), False, dtype=bool)
            

        if cum_done>0:
            av_reward /= float(cum_done)
            print('***** epsiod: {}, av_reward:{}, finish_status:{}, EPS:{} *****'.format(global_update, round(av_reward, 3),finish_status,round(1.0/cost_time,2)))
            agent_run_times=0
            for k in finish_status.keys():
                agent_run_times+=finish_status[k]
            tb_writer.add_scalar('PPO_CADRL/average_reward', av_reward, global_update)
            tb_writer.add_scalar('PPO_CADRL/reach_goal_ratio', finish_status['Reach Goal'] / agent_run_times,
                                 global_update)
            tb_writer.add_scalar('PPO_CADRL/crashed_ratio', finish_status['Crashed'] / agent_run_times,
                                 global_update)
            tb_writer.add_scalar('PPO_CADRL/time_out_ratio', finish_status['Time out'] / agent_run_times,
                                 global_update)

        # backward the policy
        avg_loss = ppo_update(params, memory, policy, optimizer)
        memory.clear()
        global_update += 1

        tb_writer.add_scalar('PPO_CADRL/average_loss', avg_loss, global_update)

        if global_update > params.pre_load_epoch and global_update % params.save_freq == 0:
            policy.save(policy_path, global_update)

        cost_time =  cost_time*0.9 + (time.time()-start_time)*0.1

if __name__ == '__main__':
    if not Config.DISCRETE_CONTROL_FLAG:
        print('pleae set Config.DISCRETE_CONTROL_FLAG to True')

    ''' define SummaryWriter '''
    log_dir ='runs/dppo_discrete_stage1/{}'.format(Config.ACT_ARCH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_writer = SummaryWriter(log_dir=log_dir)

    params = Params()
    torch.manual_seed(params.seed)

    ''' make envs '''
    train_envs = {}
    for rank in range(params.num_processes):
        train_envs[rank] = Environment(rank, True)
        # train_envs[rank].reset()

    policy = Model(params.obs_size, params.oas_size, params.action_dim, params.device,continuous_action_space=False).to(params.device)
    optimizer = optim.Adam(policy.parameters(), lr=params.lr)
    # optimizer = {'actor':optim.Adam(policy.actor.parameters(), lr=params.epoch_lr[0]),
    #              'critic': optim.Adam(policy.critic.parameters(), lr=params.epoch_lr[0])}

    policy_path = 'weight/dppo_discrete_stage1/{}'.format(Config.ACT_ARCH)
    if not os.path.exists(policy_path):
        os.makedirs(policy_path)
   
    # file = policy_path + '/{}.pth'.format(Config.NET_ARCH)
    if params.pre_load_epoch>0:
        policy.load(policy_path, params.pre_load_epoch)
    else:
        print('#####################################')
        print('############Start Training###########')
        print('#####################################')

    print('################# {} training env: 2~{} agents ####################'.format(Config.ACT_ARCH,Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))

    try:
        run(params, policy, optimizer, train_envs, policy_path,tb_writer)
    except KeyboardInterrupt:
        pass

    "tensorboard --logdir= , http://localhost:6006"

