import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from dppo.network.BaseNet import BaseNet
from layers.spikRNN import SpikingLSTM

class SpikLSTMActor(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None,continuous_action_space=True,timestep=3):
        super(SpikLSTMActor, self).__init__(obs_size,oas_size, action_space,device,timestep)

        # self.act_fea_lstm = nn.LSTM(oas_size, 256, 1, batch_first=False)
        self.act_fea_lstm = SpikingLSTM(oas_size, 512, 1)
        self.act_fc1 = nn.Linear(512, 512)

        self.act_fea_obs = nn.Linear(obs_size, 32)

        self.act_fc2 = nn.Linear(512 + 32, 256)
        self.act = nn.Linear(256, action_space)

        self.continuous_action_space=continuous_action_space

    def forward(self, x):

        host_vec, other_seq, num_other_agents = x

        N, B, _ = other_seq.shape

        num_other_agents = np.clip(num_other_agents, 1, 1000)
        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(B, 512).unsqueeze(0)

        fc1_v = fc1_s = torch.zeros(B, 512, device=self.device)
        obe_v = obe_s = torch.zeros(B, 32, device=self.device)
        fc2_v = fc2_s = torch.zeros(B, 256, device=self.device)
        act_v = act_s = torch.zeros(B, self.action_space, device=self.device)

        for step in range(self.timesteps):
            out, _ = self.act_fea_lstm(other_seq)
            a_rnn_output = out.gather(0, last_step_index_list).squeeze(0)

            fc1_v, fc1_s = self.mem_update(self.act_fc1, a_rnn_output, fc1_v, fc1_s)
            obe_v, obe_s = self.mem_update(self.act_fea_obs, host_vec, obe_v, obe_s)
            cat_s = torch.cat((obe_s, fc1_s), dim=-1)

            fc2_v, fc2_s = self.mem_update(self.act_fc2, cat_s, fc2_v, fc2_s)
            act_v, act_s = self.ns_mem_update(self.act, fc2_s, act_v, act_s)
        # value = act_v
        if self.continuous_action_space:
            output = F.tanh(act_v)
        else:
            output = F.softmax(act_v, dim=1)
        return output