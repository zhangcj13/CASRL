import random
import numpy as np
import torch
import torch.nn as nn
import os

from ddpg.network.BaseNet import BaseNet

from layers.spikRNN import SpikingLSTM

class SpikLSTM_ActorNet(BaseNet):
    """ Actor Network """
    def __init__(self, obs_size=4,oas_size=7,action_space=2,timesteps=3):
        super(SpikLSTM_ActorNet, self).__init__(obs_size,oas_size,action_space,timesteps)

        self.action_num = action_space

        self.lstm_dim = 512
        self.lstm_layer = 1
        self.lstm = SpikingLSTM(oas_size, self.lstm_dim, self.lstm_layer)

        self.act_oass_en2 = nn.Linear(self.lstm_dim, 256)

        self.act_obss_en1 = nn.Linear(obs_size, 32)

        self.layer1 = nn.Linear(256 + 32, 256)
        self.layer2 = nn.Linear(256, 256)

        self.actor1 = nn.Linear(256, 1)
        self.actor2 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=True)

        N, B, _ = other_seq.shape
        num_other_agents = np.clip(num_other_agents, 1, 1000)
        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(device=self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(B, self.lstm_dim).unsqueeze(0)

        oe2_v = oe2_s = torch.zeros(B, 256, device=self.device)
        obe_v = obe_s = torch.zeros(B, 32, device=self.device)
        ly1_v = ly1_s = torch.zeros(B, 256, device=self.device)
        ly2_v = ly2_s = torch.zeros(B, 256, device=self.device)
        ac1_v = ac1_s = torch.zeros(B, 1, device=self.device)
        ac2_v = ac2_s = torch.zeros(B, 1, device=self.device)

        for step in range(self.timesteps):
            out, _ = self.lstm(other_seq)
            encode_oass = out.gather(0, last_step_index_list).squeeze(0)
            oe2_v, oe2_s = self.mem_update(self.act_oass_en2, encode_oass, oe2_v, oe2_s)
            obe_v, obe_s = self.mem_update(self.act_obss_en1, host_vec, obe_v, obe_s)
            cat_s = torch.cat((obe_s, oe2_s), dim=-1)

            ly1_v, ly1_s = self.mem_update(self.layer1, cat_s, ly1_v, ly1_s)
            ly2_v, ly2_s = self.mem_update(self.layer2, ly1_s, ly2_v, ly2_s)

            ac1_v, ac1_s = self.ns_mem_update(self.actor1, ly2_s, ac1_v, ac1_s)
            ac2_v, ac2_s = self.ns_mem_update(self.actor2, ly2_s, ac2_v, ac2_s)

        act1 = ac1_v / self.timesteps
        act2 = ac2_v / self.timesteps

        a1 = self.sigmoid(act1)
        a2 = self.tanh(act2)

        out = torch.cat((a1, a2), dim=-1)
        return out

if __name__ == "__main__":
    model=SpikLSTM_ActorNet().cuda()

    t = np.random.rand(1, 1+26+7*6)
    t[0][1]=8
    o = model(t)

    print(o)
