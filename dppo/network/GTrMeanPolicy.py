import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from dppo.network.BaseNet import BaseNet
from layers.gtr import GTr,get_triu_attn_mask

class GTrMeanActor(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None,continuous_action_space=True):
        super(GTrMeanActor, self).__init__(obs_size,oas_size, action_space,device)

        self.act_fea_transformer = GTr(input_dim=oas_size, head_num=8, layer_num=1,embedding_dim=256,gru_gating=True)
        self.act_fc1 = nn.Linear(256, 256)
        self.act_fc2 = nn.Linear(256 + obs_size, 128)
        self.actor = nn.Linear(128, action_space)
        self.continuous_action_space=continuous_action_space

    def forward(self, x):
        host_vec, other_seq, num_other_agents = x

        N, B, _ = other_seq.shape

        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length).view(-1, 1).expand(B, 256).unsqueeze(0)

        attn_mask=get_triu_attn_mask(N, device=self.device)
        mean_vec = torch.from_numpy(np.array([1 / (n) for n in num_other_agents], dtype=np.float32)).to(
            other_seq.device)
        mean_vec = mean_vec.view(B, 1)

        # action
        trans_state = self.act_fea_transformer(other_seq, attn_mask=attn_mask)['logit']

        tran_sums = torch.zeros(N + 1, B, 256, device=self.device)
        for sl in range(N):
            tran_sums[sl + 1] = trans_state[sl] + tran_sums[sl]
        tran_sum = tran_sums.gather(0, last_step_index_list).squeeze(0)

        a_tran = tran_sum * mean_vec

        a = F.relu(self.act_fc1(a_tran))
        a = torch.cat((host_vec,a), dim=-1)
        a = F.relu(self.act_fc2(a))
        if self.continuous_action_space:
            output = F.tanh(self.actor(a))
        else:
            output = F.softmax(self.actor(a), dim=1)
        return output

class GTrMeanCritic(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None):
        super(GTrMeanCritic, self).__init__(obs_size,oas_size, action_space,device)

        self.crt_fea_transformer = GTr(input_dim=oas_size, head_num=8, layer_num=1,embedding_dim=512,gru_gating=True)
        self.crt_fc1 = nn.Linear(512, 256)
        self.crt_fc2 = nn.Linear(256 + obs_size, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        host_vec, other_seq, num_other_agents = x
        N, B, _ = other_seq.shape

        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length).view(-1, 1).expand(B, 512).unsqueeze(0)

        attn_mask = get_triu_attn_mask(N, device=self.device)
        mean_vec = torch.from_numpy(np.array([1 / (n) for n in num_other_agents], dtype=np.float32)).to(
            other_seq.device)
        mean_vec = mean_vec.view(B, 1)

        # value
        trans_state = self.crt_fea_transformer(other_seq, attn_mask=attn_mask)['logit']

        tran_sums = torch.zeros(N + 1, B, 512, device=self.device)
        for sl in range(N):
            tran_sums[sl + 1] = trans_state[sl] + tran_sums[sl]
        tran_sum = tran_sums.gather(0, last_step_index_list).squeeze(0)

        c_tran = tran_sum * mean_vec

        v = F.relu(self.crt_fc1(c_tran))
        v = torch.cat((host_vec, v), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v







