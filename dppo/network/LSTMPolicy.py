import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from dppo.network.BaseNet import BaseNet

class LSTMActor(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None,continuous_action_space=True):
        super(LSTMActor, self).__init__(obs_size,oas_size, action_space,device)

        self.act_fea_lstm = nn.LSTM(oas_size, 256, 1, batch_first=False)
        self.act_fc1 = nn.Linear(256, 256)
        self.act_fc2 = nn.Linear(256 + obs_size, 128)
        self.actor = nn.Linear(128, action_space)

        self.continuous_action_space=continuous_action_space

    def forward(self, x):

        host_vec, other_seq, num_other_agents = x

        N, B, _ = other_seq.shape

        num_other_agents = np.clip(num_other_agents, 1, 1000)
        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(B, 256).unsqueeze(0)

        # action
        ah = torch.zeros(1, B, 256).to(self.device)
        ac = torch.zeros(1, B, 256).to(self.device)

        aout, _ = self.act_fea_lstm(other_seq, (ah, ac))
        a_rnn_output = aout.gather(0, last_step_index_list).squeeze(0)

        a = F.relu(self.act_fc1(a_rnn_output))
        a = torch.cat((host_vec,a), dim=-1)
        a = F.relu(self.act_fc2(a))
        if self.continuous_action_space:
            output = F.tanh(self.actor(a))
        else:
            output = F.softmax(self.actor(a), dim=1)
        return output

class LSTMCritic(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None):
        super(LSTMCritic, self).__init__(obs_size,oas_size, action_space,device)

        self.crt_fea_lstm = nn.LSTM(oas_size, 256, 1, batch_first=False)
        self.crt_fc1 = nn.Linear(256, 256)
        self.crt_fc2 = nn.Linear(256 + obs_size, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        host_vec, other_seq, num_other_agents = x
        N, B, _ = other_seq.shape
        num_other_agents = np.clip(num_other_agents, 1, 1000)
        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(B, 256).unsqueeze(0)

        # value
        vh = torch.zeros(1, B, 256).to(self.device)
        vc = torch.zeros(1, B, 256).to(self.device)

        vout, _ = self.crt_fea_lstm(other_seq, (vh, vc))
        v_rnn_output = vout.gather(0, last_step_index_list).squeeze(0)
        v = F.relu(self.crt_fc1(v_rnn_output))
        v = torch.cat((host_vec, v), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v







