import os
import re
import numpy as np
# import tensorflow as tf
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import math
from ga3c.network.BaseNet import BaseNet

from layers.spikRNN import SpikingLSTM

class SpikLSTMNet(BaseNet):
    def __init__(self, obs_size=4, oas_size=7, num_actions=11,timesteps=5):
        super(SpikLSTMNet, self).__init__(obs_size, oas_size, num_actions,timesteps)
        self.lstm_dim = 256
        self.lstm_layer = 1

        self.lstm = SpikingLSTM(oas_size, self.lstm_dim, self.lstm_layer)

        self.obss_en1 = nn.Linear(obs_size, 32)

        self.layer1 = nn.Linear(32 + self.lstm_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.flatten = torch.nn.Flatten()

        self.fc1 = nn.Linear(256, 256)
        self.logit_v = nn.Linear(256, 1)
        self.logit_p = nn.Linear(256, num_actions)

    def forward(self, x):
        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=True)

        N, B, _ = other_seq.shape

        obe_v = obe_s = torch.zeros(B, 32, device=self.device)
        ly1_v = ly1_s = torch.zeros(B, 256, device=self.device)
        ly2_v = ly2_s = torch.zeros(B, 256, device=self.device)
        fc1_v = fc1_s = torch.zeros(B, 256, device=self.device)
        lgv_v = lgv_s = torch.zeros(B, 1, device=self.device)
        lgp_v = lgp_s = torch.zeros(B, self.num_actions, device=self.device)

        num_other_agents = np.clip(num_other_agents, 1, 1000)
        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(device=self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(B, self.lstm_dim).unsqueeze(0)

        for step in range(self.timesteps):
            out, _ = self.lstm(other_seq)
            encode_oass = out.gather(0, last_step_index_list).squeeze(0)

            obe_v, obe_s = self.mem_update(self.obss_en1, host_vec, obe_v, obe_s)
            cat_s = torch.cat((obe_s, encode_oass), dim=-1)

            ly1_v, ly1_s = self.mem_update(self.layer1, cat_s, ly1_v, ly1_s)
            ly2_v, ly2_s = self.mem_update(self.layer2, ly1_s, ly2_v, ly2_s)

            final_flat = self.flatten(ly2_s)

            fc1_v, fc1_s = self.mem_update(self.fc1, final_flat, fc1_v, fc1_s)

            lgv_v, lgv_s = self.ns_mem_update(self.logit_v, fc1_s, lgv_v, lgv_s)
            lgp_v, lgp_s = self.ns_mem_update(self.logit_p, fc1_s, lgp_v, lgp_s)

        value = torch.squeeze(lgv_v, 1)
        logit = lgp_v

        return logit, value
