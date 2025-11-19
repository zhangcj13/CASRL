import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.nn import functional as F

from layers.surrogate import pseudo_spike, DECAY
from gym_collision_avoidance.envs import Config

class BaseNet(nn.Module):
    def __init__(self, obs_size=4, oas_size=7, action_space=2, timesteps=5):
        super(BaseNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space

        self.spikefunc = pseudo_spike.apply
        self.timesteps = timesteps
        self.decay = DECAY

        if Config.NORMALIZE_INPUT:
            self.avg_vec = np.array(Config.NN_INPUT_AVG_VECTOR, dtype=np.float32)
            self.std_vec = np.array(Config.NN_INPUT_STD_VECTOR, dtype=np.float32)

    def mem_update(self, ops, x, mem, spike):
        mem = mem * self.decay * (1. - spike) + ops(x)
        spike = self.spikefunc(mem)  # act_fun : approximation firing function
        return mem, spike

    def ns_mem_update(self, ops, x, mem, spike):
        mem = mem * self.decay + ops(x)
        spike = self.spikefunc(mem)
        return mem, spike

    def forward(self, x):
        raise NotImplementedError

    def rescale_input(self, x, return_vec=False, batch_first=False, seq_cut=False):
        if Config.NORMALIZE_INPUT:
            x_normalized = (x[:, 1:] - self.avg_vec) / self.std_vec
        else:
            x_normalized = x[:, 1:]

        host_agent_vec = x_normalized[:,
                         Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        host_agent_vec = torch.Tensor(host_agent_vec).to(self.device)

        if Config.USING_LASER:
            laser_scan = x_normalized[:, Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
            laser_scan = torch.Tensor(laser_scan).to(self.device)
            return host_agent_vec, laser_scan

        num_other_agents = np.clip(x_normalized[:, 0] + 1, 0, 1e6)
        other_agent_vec = x_normalized[:, Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        other_agent_vec = torch.Tensor(other_agent_vec).to(self.device)
        if return_vec:
            return host_agent_vec, other_agent_vec, num_other_agents
        other_agent_seq = torch.reshape(other_agent_vec, [-1, Config.MAX_NUM_OTHER_AGENTS_OBSERVED,
                                                          Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])

        if seq_cut:
            max_effective_length = np.max(num_other_agents).astype(np.int)
            max_effective_length = max_effective_length if max_effective_length >= 1 else 1
            other_agent_seq = other_agent_seq[:, :max_effective_length, :]
        if not batch_first:
            other_agent_seq = torch.transpose(other_agent_seq, 0, 1)
        return host_agent_vec, other_agent_seq, num_other_agents

    def rescale_scan_input(self, x, to_torch=False):
        if Config.NORMALIZE_INPUT:
            x_normalized = (x[:, 1:] - self.avg_vec) / self.std_vec
        else:
            x_normalized = x[:, 1:]

        host_agent_vec = x_normalized[:,
                         Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        laser_scan = x_normalized[:, Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        if to_torch:
            host_agent_vec = torch.Tensor(host_agent_vec).to(self.device)
            laser_scan = torch.Tensor(laser_scan).to(self.device)
        return host_agent_vec, laser_scan
    def reset_mean_std(self):
        if Config.NORMALIZE_INPUT:
            self.avg_vec = np.array(Config.NN_INPUT_AVG_VECTOR, dtype=np.float32)
            self.std_vec = np.array(Config.NN_INPUT_STD_VECTOR, dtype=np.float32)

if __name__ == '__main__':
    from torch.autograd import Variable

    # net = MLPPolicy(3, 2)
    #
    # observation = Variable(torch.randn(2, 3))
    # v, action, logprob, mean = net.forward(observation)
    # print(v)
    batch = 5
    x = Variable(torch.randn(batch, 3, 512))
    goal = Variable(torch.randn(batch, 2))
    speed = Variable(torch.randn(batch, 2))
    # spk_x = Variable(torch.randn(batch, 3, 512))

    spk_x = [Variable(torch.randn(batch, 1, 512)),
             Variable(torch.randn(batch, 1, 512)),
             Variable(torch.randn(batch, 1, 512))]

    spk_goal = Variable(torch.randn(batch, 3))
    spk_speed = Variable(torch.randn(batch, 3))

    # net = SpikingCNNPolicy3C_Act(frames=3, action_space=2)
    # v, action, logprob, mean = net.forward(x, goal, speed, spk_x,spk_goal,spk_speed,batch_size=batch)
    # print(v)







