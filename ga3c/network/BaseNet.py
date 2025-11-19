import os
import re
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import math

from gym_collision_avoidance.envs import Config
from layers.surrogate import pseudo_spike, DECAY


class BaseNet(nn.Module):
    def __init__(self, obs_size=4, oas_size=7, num_actions=11, timesteps=5):
        super(BaseNet, self).__init__()
        self.spike_func = pseudo_spike.apply
        self.timesteps = timesteps
        self.decay = DECAY

        if Config.NORMALIZE_INPUT:
            self.avg_vec = np.array(Config.NN_INPUT_AVG_VECTOR, dtype=np.float32)
            self.std_vec = np.array(Config.NN_INPUT_STD_VECTOR, dtype=np.float32)

        self.device = torch.device(Config.DEVICE)

        self.num_actions = num_actions

        self.y_r = 0
        self.log_epsilon = Config.LOG_EPSILON
        self.var_beta = Config.BETA_START

    def mem_update(self, ops, x, mem, spike):
        mem = mem * self.decay * (1. - spike) + ops(x)
        spike = self.spike_func(mem)  # act_fun : approximation firing function
        return mem, spike

    def ns_mem_update(self, ops, x, mem, spike):
        mem = mem * self.decay + ops(x)
        spike = self.spike_func(mem)
        return mem, spike

    def forward(self, x):
        raise NotImplementedError

    def evaluate_actions(self, x, y_r, a):
        y_r_t = torch.from_numpy(y_r).to(self.device)
        a_t = torch.from_numpy(a).to(self.device)
        logits_p, logits_v = self.forward(x)

        # Cost: v
        advantage = y_r_t - logits_v
        cost_v = 0.5 * advantage.pow(2)
        cost_v = cost_v.sum(dim=-1, keepdim=True)

        # Cost: p
        prob = F.softmax(logits_p, dim=-1)
        log_prob = F.log_softmax(logits_p, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=False)

        log_selected_action_prob = (log_prob * a_t).sum(1, keepdim=False)

        cost_p_advant_agg = log_selected_action_prob * (y_r_t - logits_v.detach())
        cost_p_entrop_agg = -1 * self.var_beta * entropy

        cost_p_advant_agg = cost_p_advant_agg.sum(0)
        cost_p_entrop_agg = cost_p_entrop_agg.sum(0)

        cost_p = -(cost_p_advant_agg + cost_p_entrop_agg)

        cost_all = cost_p + cost_v

        return cost_all

        # v, _, _, mean = self.forward(x, goal, speed)
        # logstd = self.logstd.expand_as(mean)
        # std = torch.exp(logstd)
        # # evaluate
        # logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        # dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        # dist_entropy = dist_entropy.sum(-1).mean()
        # return v, logprob, dist_entropy

    def rescale_input(self, x_normalized, return_vec=False, batch_first=False, seq_cut=False):
        host_agent_vec = x_normalized[:,
                         Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        host_agent_vec = torch.Tensor(host_agent_vec).to(self.device)

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
