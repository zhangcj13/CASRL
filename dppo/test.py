# test

import os
import sys
import gym
from gym import wrappers
import time
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import Model

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def test(rank, params, shared_model, shared_obs_stats, test_n, env):
    torch.manual_seed(params.seed + rank)
    work_dir = mkdir('runs', 'dppo')
    monitor_dir = mkdir(work_dir, 'monitor')
    # env = gym.make(params.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    # num_inputs = env.observation_space.shape[0]
    # num_outputs = env.action_space.shape[0]
    # model = Model(num_inputs, num_outputs)
    model = Model(params.obs_size, params.oas_size, params.action_dim,params.device).to(params.device)

    # state = env.reset()
    # state = Variable(torch.Tensor(state).unsqueeze(0))
    env.reset()
    # reward_sum = 0
    game_over = True
    reward_sum_loggers = np.zeros((params.max_num_agents))

    start_time = time.time()

    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        # shared_obs_stats.observes(state)
        #print(shared_obs_stats.n[0])
        # state = shared_obs_stats.normalize(state)

        env_actions = {}
        actions = {}
        for i, agent_observation in enumerate(env.latest_observations):
            is_agent_running = agent_observation[0]
            if not is_agent_running:
                continue

            mu, sigma_sq, _ = model(np.asarray([agent_observation]))
            eps = torch.randn(mu.size()).to(params.device)
            action = mu + sigma_sq.sqrt() * Variable(eps)
            env_action = action.data.cpu().squeeze().numpy()

            env_action = np.clip(env_action, a_min=params.action_bound[0], a_max=params.action_bound[1])

            actions[i] = env_action
            env_actions[i] = env_action

        rewards, game_over, infos = env.step([actions], rank, episode_length, ampl2goal=params.ampl2goal)
        rewards = rewards[0]
        # reward_sum += reward

        which_agents_done = infos[0]['which_agents_done']
        which_agents_learning = infos[0]['which_agents_learning']
        num_agents_running_ga3c = np.sum(list(which_agents_learning.values()))
        which_agents_status_dict = infos[0]['which_agents_status_dict']

        for i in which_agents_learning.keys():
            # Loop through all feedback from environment (which may not be equal to Config.MAX_NUM_AGENTS)
            if not which_agents_learning[i]:
                continue

            reward_sum_loggers[i] += rewards[i]

        if game_over:
            ave_reward=np.sum(reward_sum_loggers)/num_agents_running_ga3c
            print("Time {}, episode ave_reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                ave_reward, episode_length))
            reward_sum_loggers = np.zeros((params.max_num_agents))
            episode_length = 0
            env.reset()
            time.sleep(10)

        # state = Variable(torch.Tensor(state).unsqueeze(0))
