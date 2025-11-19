import random
import numpy as np
import torch
import torch.nn as nn
import os

from ddpg.network.BaseNet import BaseNet
from layers.gtr import GTr

class GTr_ActorNet(BaseNet):
    """ Actor Network """
    def __init__(self, obs_size=4,oas_size=7,action_space=2):
        super(GTr_ActorNet, self).__init__(obs_size, oas_size, action_space)

        GRU_GATING = True
        Head_num = 4
        TrLayer_num = 1

        self.transformer = GTr(input_dim=oas_size, head_num=Head_num, layer_num=TrLayer_num,embedding_dim=256,gru_gating=GRU_GATING)

        self.fc2 = nn.Linear(256 + obs_size, 256)
        self.fc3 = nn.Linear(256, 256)

        self.actor1 = nn.Linear(256, 1)
        self.actor2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=False)

        N, B, _ = other_seq.shape

        key_padding_mask = []
        for i in range(B):
            for j in range(N):
                key_padding_mask.append(0 if j < num_other_agents[i] else 1)
        key_padding_mask = torch.from_numpy(np.array(key_padding_mask, dtype=np.float32)).view(B,N).to(device=self.device)

        trans_state = self.transformer(other_seq,attn_mask=key_padding_mask)['logit']
        a0=torch.mean(trans_state,0)

        a = torch.cat((host_vec, a0), dim=-1)
        a = self.relu(self.fc2(a))
        a = self.relu(self.fc3(a))

        a1 = self.sigmoid(self.actor1(a))
        a2 = self.tanh(self.actor2(a))
        out = torch.cat((a1, a2), dim=-1)
        return out

class GTr_CriticNet(BaseNet):
    """ Critic Network"""
    def __init__(self, obs_size=4,oas_size=2,action_space=2):
        super(GTr_CriticNet, self).__init__(obs_size,oas_size,action_space)

        GRU_GATING = True
        Head_num = 8
        TrLayer_num = 1

        self.transformer = GTr(input_dim=oas_size, head_num=Head_num, layer_num=TrLayer_num,embedding_dim=512,gru_gating=GRU_GATING)

        self.fc2 = nn.Linear(512 + obs_size + action_space, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, xa):
        x, a = xa

        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=True)
        N, B, _ = other_seq.shape

        key_padding_mask = []
        for i in range(B):
            for j in range(N):
                key_padding_mask.append(0 if j < num_other_agents[i] else 1)
        key_padding_mask = torch.from_numpy(np.array(key_padding_mask, dtype=np.float32)).view(B, N).to(
            device=self.device)

        trans_state = self.transformer(other_seq, attn_mask=key_padding_mask)['logit']
        v0=torch.mean(trans_state,0)

        v = torch.cat((host_vec, v0, a), dim=-1)

        v = self.relu(self.fc2(v))
        v = self.relu(self.fc3(v))
        out = self.fc4(v)
        return out


if __name__ == "__main__":
    model=GTr_ActorNet().cuda()

    t = np.random.rand(1, 1+26+7*0)
    t[0][1]=2
    o = model(t)

    # o=GPTConfig()

    print(o)
