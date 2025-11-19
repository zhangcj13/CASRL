import os
# os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase_DDPG'
import random
import numpy as np
import torch
import torch.nn as nn

from ddpg.network.BaseNet import BaseNet
from layers.spikGTr import SpikGTr

class SpikGTr_ActorNet(BaseNet):
    """ Actor Network """
    def __init__(self, obs_size=4,oas_size=7,action_space=2,timesteps=3):
        super(SpikGTr_ActorNet, self).__init__(obs_size, oas_size, action_space,timesteps)

        GRU_GATING = True
        Head_num = 8
        TrLayer_num = 1

        self.transformer = SpikGTr(input_dim=oas_size, head_num=Head_num, layer_num=TrLayer_num,embedding_dim=512,gru_gating=GRU_GATING,device=self.device)

        self.act_oass_en2 = nn.Linear(512, 256)
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
        self.transformer.set_v(B, N, other_seq.device)

        key_padding_mask = []
        for i in range(B):
            for j in range(N):
                key_padding_mask.append(0 if j < num_other_agents[i] else 1)
        key_padding_mask = torch.from_numpy(np.array(key_padding_mask, dtype=np.float32)).view(B,N).to(device=self.device)

        oe2_v = oe2_s = torch.zeros(B, 256, device=self.device)
        obe_v = obe_s = torch.zeros(B, 32, device=self.device)
        ly1_v = ly1_s = torch.zeros(B, 256, device=self.device)
        ly2_v = ly2_s = torch.zeros(B, 256, device=self.device)
        ac1_v = ac1_s = torch.zeros(B, 1, device=self.device)
        ac2_v = ac2_s = torch.zeros(B, 1, device=self.device)

        for step in range(self.timesteps):
            trans_state = self.transformer(other_seq,attn_mask=key_padding_mask)['logit']
            a0 = torch.mean(trans_state, 0)

            oe2_v, oe2_s = self.mem_update(self.act_oass_en2, a0, oe2_v, oe2_s)
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
    model=SpikGTr_ActorNet().cuda()

    t = np.random.rand(1, 1+26+7*6)
    t[0][1]=8
    o = model(t)

    print(o)
