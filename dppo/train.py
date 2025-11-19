import os
import sys
import numpy as np
import random
import gym
import math
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp

from dppo.model import Model

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []
    def __len__(self,):
        return len(self.memory)

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))

        return map(lambda x: np.concatenate(x, axis=0), samples)

        # batch_data=[]
        # for i, data in enumerate(samples):
        #     if i==0:
        #         bd = np.concatenate(data, axis=0)
        #     else:
        #         bd = torch.cat(data, 0)
        #     batch_data.append(bd)
        #
        # return batch_data

        # return map(lambda x: torch.cat(x, 0) if type(x) == torch.Tensor else np.concatenate(x, axis=0), samples)
        # return map(lambda x: torch.cat(x, 0), samples)
    def get_mean_std(self,index):
        dlst=[]
        for m in self.memory:
            dlst.append(m[index])
        dlst=np.asarray(dlst)
        mean = dlst.mean()
        std = dlst.std()
        # for m in self.memory:
        #     m[index]= (m[index]-mean) /std
        return mean,std

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            pass
        shared_param._grad = param.grad

def normal(x, mu, std):
    a = (-1*(x-mu).pow(2)/(2*std)).exp()
    b = 1/(2*std*np.pi).sqrt()
    return a*b

def train(rank, params, traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats, test_n,env):
    torch.manual_seed(params.seed)
    num_outputs = params.action_dim
    model = Model(params.obs_size, params.oas_size, params.action_dim,params.device).to(params.device)

    memory = ReplayMemory(params.exploration_size)

    env.reset()
    game_over = True
    episode_length = 0

    # buff for training
    states = {i: [] for i in range(params.max_num_agents)}
    actions = {i: [] for i in range(params.max_num_agents)}
    rewards = {i: [] for i in range(params.max_num_agents)}
    values = {i: [] for i in range(params.max_num_agents)}
    returns = {i: [] for i in range(params.max_num_agents)}
    advantages = {i: [] for i in range(params.max_num_agents)}
    targets = {i: [] for i in range(params.max_num_agents)}
    logprobs = {i: [] for i in range(params.max_num_agents)}
    dones = {i: [] for i in range(params.max_num_agents)}

    step_counts = np.zeros((params.max_num_agents))
    cur_which_agents_done = np.full((params.max_num_agents), False, dtype=bool)
    which_agents_done = np.full((params.max_num_agents), False, dtype=bool)
    which_agents_done_and_trained = np.full((params.max_num_agents), False, dtype=bool)
    which_agents_learning = np.full((params.max_num_agents), True, dtype=bool)

    cum_reward_sum_logger = np.zeros((params.max_num_agents))

    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())

        w = -1
        av_reward = 0
        cum_done = 0
        nb_runs = 0
        reward_0 = 0
        t = -1
        while w < params.exploration_size:
            t+=1
            # # Perform K steps
            # for step in range(params.num_steps):
            # Perform every agent's step<num_steps and no agent is done
            while np.max(step_counts)<params.num_steps and not True in cur_which_agents_done:
                w+=1
                # shared_obs_stats.observes(state)
                # state = shared_obs_stats.normalize(state)

                env_actions={}
                for i, agent_observation in enumerate(env.latest_observations):
                    is_agent_running = agent_observation[0]
                    if not is_agent_running or which_agents_done[i]:
                        continue

                    state = np.asarray([agent_observation])
                    v, a, logprob, scaled_action = model.generate_action(state, params.action_bound)

                    env_actions[i] = scaled_action[0]
                    states[i].append(state)
                    actions[i].append(a)
                    values[i].append(v)
                    logprobs[i].append(logprob)

                cur_rewards, game_over, infos = env.step([env_actions], rank, w, ampl2goal=params.ampl2goal)
                cur_rewards = cur_rewards[0]

                which_agents_done = infos[0]['which_agents_done']
                which_agents_learning = infos[0]['which_agents_learning']
                num_agents_running_ga3c = np.sum(list(which_agents_learning.values()))
                # which_agents_status_dict = infos[0]['which_agents_status_dict']

                for i in which_agents_learning.keys():
                    # Loop through all feedback from environment (which may not be equal to Config.MAX_NUM_AGENTS)
                    if not which_agents_learning[i] or which_agents_done_and_trained[i]:
                        continue
                    reward = cur_rewards[i]
                    done = which_agents_done[i]
                    cur_which_agents_done[i]=done
                    dones[i].append(done)
                    cum_reward_sum_logger[i] += cur_rewards[i]
                    rewards[i].append(np.array([[reward]]))

                    step_counts[i]+=1

                if game_over:
                    cum_done += 1
                    av_reward += np.sum(cum_reward_sum_logger)/num_agents_running_ga3c
                    # cum_reward = 0
                    # av_reward_sum_logger += cum_reward_sum_logger
                    cum_reward_sum_logger = np.zeros((params.max_num_agents))
                    episode_length = 0

                    env.reset()
                    cur_which_agents_done = np.full((params.max_num_agents), False, dtype=bool)
                    which_agents_done = np.full((params.max_num_agents), False, dtype=bool)
                    which_agents_done_and_trained = np.full((params.max_num_agents), False, dtype=bool)
                    which_agents_learning = np.full((params.max_num_agents), True, dtype=bool)

            for n in which_agents_learning.keys():
                if not which_agents_learning[n] or which_agents_done_and_trained[n]:
                    continue

                if step_counts[n]>=params.num_steps or cur_which_agents_done[n]:
                    # one last step
                    R = np.zeros((1, 1))
                    if not which_agents_done[n]:
                        agent_observation = env.latest_observations[n]
                        v, _, _, _ = model.generate_action(np.asarray([agent_observation]),params.action_bound)
                        R = v
                    else:
                        which_agents_done_and_trained[n] = True
                    values[n].append(R)
                    # compute returns and GAE(lambda) advantages:
                    num_step= len(rewards[n])
                    gae = np.zeros((1,1))
                    for t in range(num_step - 1, -1, -1):
                        delta = rewards[n][t] + params.gamma * values[n][t + 1] - values[n][t]
                        gae = delta + params.gamma * params.gae_param  * gae
                        advantages[n].insert(0, gae)
                        R = gae + values[n][t]
                        returns[n].insert(0, R)

                    memory.push([states[n], actions[n],logprobs[n], returns[n], values[n],rewards[n],advantages[n]])

                    states[n] = []
                    actions[n] = []
                    logprobs[n] = []
                    returns[n] = []
                    rewards[n] = []
                    values[n] = []
                    advantages[n] = []

                    step_counts[n] = 0
                    cur_which_agents_done[n]=False

        # policy grad updates:
        av_reward /= float(cum_done+1)
        model_old = Model(params.obs_size, params.oas_size, params.action_dim, params.device).to(params.device)
        model_old.load_state_dict(model.state_dict())
        if t==0:
            reward_0 = av_reward-(1e-2)
        #batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(params.batch_size)
        for k in range(params.num_epoch):
            # load new model
            model.load_state_dict(shared_model.state_dict())
            model.zero_grad()
            # get initial signal
            signal_init = traffic_light.get()
            # new mini_batch
            sampled_states, sampled_actions, sampled_logprobs, sampled_returns,sampled_values, sampled_rewards, sampled_advantages = memory.sample(params.batch_size)

            sampled_actions = Variable(torch.from_numpy(sampled_actions)).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(sampled_logprobs)).float().cuda()
            sampled_returns = Variable(torch.from_numpy(sampled_returns)).float().cuda()
            sampled_advantages = Variable(torch.from_numpy(sampled_advantages)).float().cuda()

            new_value, new_logprob, dist_entropy = model.evaluate_actions(sampled_states, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advantages.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - params.clip, 1 + params.clip) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_returns.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - params.coeff_entropy * dist_entropy

            loss.backward()

            # prepare for step
            # total_loss.backward(retain_graph=True)
            #ensure_shared_grads(model, shared_model)
            #shared_model.cum_grads()
            shared_grad_buffers.add_gradient(model)

            counter.increment()

            # wait for a new signal to continue
            while traffic_light.get() == signal_init:
                pass

        test_n += 1
        memory.clear()
