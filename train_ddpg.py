import time
import os
os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase_DDPG'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import numpy as np

import gym
gym.logger.set_level(40)
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper, MultiagentFlattenDictWrapper, MultiagentDummyVecEnv, MultiagentDictToMultiagentArrayWrapper
from gym_world import gym_world

from ddpg.agent import Agent

try:
    import torch
    import torch.nn as nn
    from utils import creat_training_agents
    from torch.optim import Adam
    from torch.utils.tensorboard import SummaryWriter
except:
    print("main Torch not installed...")


def reset_env(num_agents=2,plot_dir=None):
    Config.set_agents_num(num_agents)

    env = gym_world()
    env = MultiagentDictToMultiagentArrayWrapper(env, dict_keys=Config.STATES_IN_OBS, max_num_agents=num_agents)
    if plot_dir is not None:
        env.set_plot_save_dir(plot_dir)

    # create sim agent
    env_range = ((-Config.CIRCLE_RADIUS, Config.CIRCLE_RADIUS), (-Config.CIRCLE_RADIUS, Config.CIRCLE_RADIUS))
    env_agents = creat_training_agents(num_agents, '',
                                       radius=0.2, pref_speed=1.0,
                                       env_range=env_range, poly_raw_list=[], goal_raw_list=[], diff=2.0,
                                       r_circle=2,
                                       policy='CNNOASS', sensor=Config.STATES_IN_OBS[-1])
    env.set_agents(env_agents)
    return env

def train(episode_num=(100, 200, 300, 400),
               iteration_num_start=(1000, 300, 400, 500), iteration_num_step=(1, 2, 3, 4),
               iteration_num_max=(1000, 1000, 1000, 1000),
               iteration_agents_num=(2, 4, 6,10),
               save_steps=10000,
               env_epsilon=(0.9, 0.6, 0.6, 0.6), env_epsilon_decay=(0.999, 0.9999, 0.9999, 0.9999),
               state_num=4, action_num=2,
               memory_size=10000, batch_size=16, epsilon_end=0.1, rand_start=10000, rand_decay=0.999,
               rand_step=2, target_tau=0.005, target_step=1, use_cuda=True,
               weight_pth=None,oas_num=None):
    # Create Folder to save weights
    dirName = Config.ACT_ARCH
    try:
        os.mkdir('./weight/ddpg/' + dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    # Define first training environment
    num_agents = iteration_agents_num[0]
    plot_dir=os.path.dirname(os.path.realpath(__file__)) + '/GCA_Plot/train/{}/'.format(Config.ACT_ARCH)
    env=reset_env(num_agents, plot_dir=plot_dir)

    # create train agent
    rescale_state_num = state_num
    agent = Agent(state_num, action_num, rescale_state_num,
                      memory_size=memory_size, batch_size=batch_size, epsilon_end=epsilon_end,
                      epsilon_rand_decay_start=rand_start, epsilon_decay=rand_decay, epsilon_rand_decay_step=rand_step,
                      target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda,oas_num=oas_num,other_agents_num=Config.AGENTS_STATES_TO_SIM)

    if weight_pth is not None:
        agent.load(weight_pth)

    # st=31
    # act_file_name = './weight/ddpg/{}/{}_s{}.pth'.format(Config.ACT_ARCH, Config.ACT_ARCH, st)
    # crt_file_name = './weight/ddpg/{}/{}_s{}.pth'.format(Config.ACT_ARCH, Config.CRT_ARCH, st)
    # agent.load(act_file_name,crt_file_name)

    # Define Tensorboard Writer
    tb_writer = SummaryWriter()

    # Define maximum steps per episode and reset maximum random action
    NUM_ENV = num_agents
    ITA_NUM=len(episode_num)

    overall_steps = 0
    overall_episode = 0
    env_episode = 0
    env_ita = 0

    agent.reset_epsilon(env_epsilon[env_ita],
                        env_epsilon_decay[env_ita])

    epnum = np.ones((NUM_ENV)) * episode_num[env_ita]
    ep_reward = np.zeros((NUM_ENV))
    terminal_list = np.zeros((NUM_ENV)) > 1e6
    steps = np.ones((NUM_ENV))
    max_step=Config.TERMINAL_STEP

    # Start Training
    start_time = time.time()
    state = env.reset()

    while np.min(epnum) > 0 :
        idxs = []
        for id, terminal in enumerate(terminal_list):
            if terminal:
                terminal_list[id] = False
                ep_reward[id] = 0
                steps[id] = 0
                epnum[id] -= 1
                idxs.append(id)
        state=env.reset_agent(idxs,num_agents)
        state=env.observation(state)
        rescale_state = np.asarray(state)

        while not True in terminal_list:
            ita_time_start = time.time()
            overall_steps += 1

            raw_action = agent.act(rescale_state)
            decode_action =agent.action_decoder(raw_action)

            for i in range(Config.MAX_NUM_AGENTS_TO_SIM - Config.AGENTS_TO_Train):
                decode_action.append([0.0, 0.0])

            next_state, rewards, game_over, which_agents_done = env.step(decode_action)

            terminal_list, results = env.is_terminal()
            done=terminal_list[0:Config.AGENTS_TO_Train]
            train_rewards=rewards[0:Config.AGENTS_TO_Train]
            ep_reward += train_rewards

            rescale_next_state = np.asarray(next_state)

            # Add a last step negative reward
            agent.remember(state, rescale_state, raw_action, train_rewards, next_state, rescale_next_state, done)
            state = next_state
            rescale_state = rescale_next_state

            # Train network with replay
            if len(agent.memory) > batch_size:
                actor_loss_value, critic_loss_value = agent.replay()
                tb_writer.add_scalar('MACA/actor_loss', actor_loss_value, overall_steps)
                tb_writer.add_scalar('MACA/critic_loss', critic_loss_value, overall_steps)
            ita_time_end = time.time()
            tb_writer.add_scalar('MACA/ita_time', ita_time_end - ita_time_start, overall_steps)
            tb_writer.add_scalar('MACA/action_epsilon', agent.epsilon, overall_steps)

            # Save Model
            if overall_steps % save_steps == 0:
                agent.save("./weight/ddpg/" + dirName, overall_steps // save_steps)

            # If Done then break
            if True in done or np.max(steps)>=max_step :
                avg_reward=np.average(ep_reward)
                print("Episode: {}/{}, step: {}, Avg Reward: {}".format(overall_episode, episode_num, overall_steps,avg_reward))
                tb_writer.add_scalar('MACA/avg_reward', avg_reward, overall_steps)
                break
            steps += 1

        if overall_episode == 999:
            agent.save("./weight/ddpg/"+ dirName, 0)
        overall_episode += 1
        env_episode += 1
        if env_episode == episode_num[env_ita]:
            print("Environment ", env_ita, " Training Finished ...")

            env_ita += 1
            if env_ita>=ITA_NUM:
                break
            num_agents = iteration_agents_num[env_ita]
            env = reset_env(num_agents, plot_dir=plot_dir)
            NUM_ENV=num_agents
            state = env.reset()
            agent.reset_epsilon(env_epsilon[env_ita],
                                env_epsilon_decay[env_ita])
            epnum = np.ones((NUM_ENV)) * episode_num[env_ita]
            ep_reward = np.zeros((NUM_ENV))
            terminal_list = np.zeros((NUM_ENV)) > 1e6
            steps = np.ones((NUM_ENV))
            agent.reset_status()
            env_episode = 0

            agent.save("./weight/ddpg/episode_" + dirName, env_ita)

    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False

    episode_num = (500, 1500, 8000, 90000)
    # episode_num = (300, 900, 3000, 10000)
    # episode_num = (5, 20, 50, 100000)
    # train(use_cuda=USE_CUDA, state_num=4, oas_num=7, episode_num=episode_num)
    train(use_cuda=USE_CUDA, state_num=4, oas_num=36, episode_num=episode_num)
