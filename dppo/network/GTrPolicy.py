import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from dppo.network.BaseNet import BaseNet
from layers.gtr import GTr

class GTrActor(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None,continuous_action_space=True):
        super(GTrActor, self).__init__(obs_size,oas_size, action_space,device)

        self.act_fea_transformer = GTr(input_dim=oas_size, head_num=8, layer_num=1,embedding_dim=256,gru_gating=True)
        self.act_fc1 = nn.Linear(256, 256)
        self.act_fc2 = nn.Linear(256 + obs_size, 128)
        self.actor = nn.Linear(128, action_space)

        self.continuous_action_space=continuous_action_space

    def forward(self, x):

        host_vec, other_seq, num_other_agents = x

        N, B, _ = other_seq.shape

        key_padding_mask = []
        for i in range(B):
            for j in range(N):
                key_padding_mask.append(0 if j < num_other_agents[i] else 1)
        key_padding_mask = torch.from_numpy(np.array(key_padding_mask, dtype=np.float32)).view(B, N).to(
            device=self.device)
        # action
        trans_state = self.act_fea_transformer(other_seq, key_padding_mask=key_padding_mask)['logit']
        a_tran = torch.mean(trans_state, 0)

        a = F.relu(self.act_fc1(a_tran))
        a = torch.cat((host_vec,a), dim=-1)
        a = F.relu(self.act_fc2(a))
        if self.continuous_action_space:
            output = F.tanh(self.actor(a))
        else:
            output = F.softmax(self.actor(a), dim=1)
        return output

class GTrCritic(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None):
        super(GTrCritic, self).__init__(obs_size,oas_size, action_space,device)

        self.crt_fea_transformer = GTr(input_dim=oas_size, head_num=8, layer_num=1,embedding_dim=512,gru_gating=True)
        self.crt_fc1 = nn.Linear(512, 256)
        self.crt_fc2 = nn.Linear(256 + obs_size, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        host_vec, other_seq, num_other_agents = x
        N, B, _ = other_seq.shape

        key_padding_mask = []
        for i in range(B):
            for j in range(N):
                key_padding_mask.append(0 if j < num_other_agents[i] else 1)
        key_padding_mask = torch.from_numpy(np.array(key_padding_mask, dtype=np.float32)).view(B, N).to(
            device=self.device)

        # value
        trans_state = self.crt_fea_transformer(other_seq, key_padding_mask=key_padding_mask)['logit']
        c_tran = torch.mean(trans_state, 0)

        v = F.relu(self.crt_fc1(c_tran))
        v = torch.cat((host_vec, v), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v







