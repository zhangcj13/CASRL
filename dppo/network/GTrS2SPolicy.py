import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from dppo.network.BaseNet import BaseNet
from layers.gtr import GTr,get_key_padding_mask,get_triu_attn_mask

class GTrS2SActor(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None,continuous_action_space=True):
        super(GTrS2SActor, self).__init__(obs_size,oas_size, action_space,device)

        self.act_fea_transformer = GTr(input_dim=oas_size, head_num=8, layer_num=1,embedding_dim=256,gru_gating=True)
        self.act_fc1 = nn.Linear(256, 256)
        self.act_fc2 = nn.Linear(256 + obs_size, 128)
        self.actor = nn.Linear(128, action_space)

        self.continuous_action_space=continuous_action_space

    def forward(self, x):

        host_vec, other_seq, num_other_agents = x

        N, B, _ = other_seq.shape

        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(B, 256).unsqueeze(0)

        attn_mask = get_triu_attn_mask(N,self.device)
        key_padding_mask = get_key_padding_mask(B, N, num_other_agents, device=self.device)
        # action
        trans_state = self.act_fea_transformer(other_seq,attn_mask=attn_mask, key_padding_mask=key_padding_mask)['logit']

        a_tran = trans_state.gather(0, last_step_index_list).squeeze(0)
        # a_tran = torch.mean(trans_state, 0)

        a = F.relu(self.act_fc1(a_tran))
        a = torch.cat((host_vec,a), dim=-1)
        a = F.relu(self.act_fc2(a))
        if self.continuous_action_space:
            output = F.tanh(self.actor(a))
        else:
            output = F.softmax(self.actor(a), dim=1)
        return output

class GTrS2SCritic(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None):
        super(GTrS2SCritic, self).__init__(obs_size,oas_size, action_space,device)

        self.crt_fea_transformer = GTr(input_dim=oas_size, head_num=8, layer_num=1,embedding_dim=512,gru_gating=True)
        self.crt_fc1 = nn.Linear(512, 256)
        self.crt_fc2 = nn.Linear(256 + obs_size, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        host_vec, other_seq, num_other_agents = x
        N, B, _ = other_seq.shape

        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(B, 512).unsqueeze(0)

        attn_mask = get_triu_attn_mask(N, self.device)
        key_padding_mask = get_key_padding_mask(B, N, num_other_agents, device=self.device)

        # value
        trans_state = self.crt_fea_transformer(other_seq,attn_mask=attn_mask, key_padding_mask=key_padding_mask)['logit']
        # c_tran = torch.mean(trans_state, 0)
        c_tran = trans_state.gather(0, last_step_index_list).squeeze(0)

        v = F.relu(self.crt_fc1(c_tran))
        v = torch.cat((host_vec, v), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v







