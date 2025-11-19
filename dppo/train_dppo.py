import os
os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase_PPOS'

import argparse
import sys
import gym
gym.logger.set_level(40)
from gym_collision_avoidance.envs import Config
# Config.set_display()
from ga3c.Environment import Environment

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from dppo.model import Model, Shared_grad_buffers, Shared_obs_stats
from dppo.train import train
from dppo.test import test
from dppo.chief import chief
from dppo.utils import TrafficLight, Counter

class Params():
    def __init__(self):
        self.batch_size = 1000
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 100 #1000
        self.exploration_size = 1000
        self.num_processes = 1 # 4
        self.update_treshold = self.num_processes - 1
        self.max_episode_length = 10000
        self.seed = 1
        self.env_name = '"CollisionAvoidance-v0"'

        self.obs_size = Config.HOST_AGENT_OBSERVATION_LENGTH = 4
        self.oas_size = Config.OTHER_AGENT_OBSERVATION_LENGTH
        self.action_dim = 2

        self.action_bound = [[-1, -1], [1, 1]]
        self.ampl2goal = 0.08
        self.max_num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT

        self.coeff_entropy=0.02

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    params = Params()
    torch.manual_seed(params.seed)

    ''' make envs '''
    train_envs = {}
    for rank in range(params.num_processes):
        train_envs[rank] = Environment(rank, True)
    test_env=Environment(params.num_processes, True)

    traffic_light = TrafficLight()
    counter = Counter()

    shared_model = Model(params.obs_size, params.oas_size, params.action_dim,params.device).to(params.device)
    shared_model.share_memory()
    shared_grad_buffers = Shared_grad_buffers(shared_model)
    #shared_grad_buffers.share_memory()
    ''' need edit '''
    shared_obs_stats = Shared_obs_stats(params.obs_size)
    ''' need edit '''
    #shared_obs_stats.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=params.lr)
    test_n = torch.Tensor([0])
    test_n.share_memory_()

    train(0, params, traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats, test_n,train_envs[0])

    #
    # processes = []
    # p = mp.Process(target=test, args=(params.num_processes, params, shared_model, shared_obs_stats, test_n,test_env))
    # # p.start()
    # p.run()
    # processes.append(p)
    # p = mp.Process(target=chief, args=(params.num_processes, params, traffic_light, counter, shared_model, shared_grad_buffers, optimizer))
    # # p.start()
    # p.run()
    # processes.append(p)
    # for rank in range(0, params.num_processes):
    #     p = mp.Process(target=train, args=(rank, params, traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats, test_n,train_envs[rank]))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
