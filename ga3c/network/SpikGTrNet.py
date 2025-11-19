import os
os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase1_GA3C'
import numpy as np
import torch
import torch.nn as nn

from ga3c.network.BaseNet import BaseNet

from layers.spikGTr import SpikGTr


class SpikGTrNet(BaseNet):
    def __init__(self, obs_size=4, oas_size=7, num_actions=11,timesteps=5):
        super(SpikGTrNet, self).__init__(obs_size, oas_size, num_actions,timesteps)
        self.num_actions = num_actions

        GRU_GATING = True
        Head_num = 8  # 4
        TrLayer_num = 1  # 1

        self.transformer = SpikGTr(input_dim=oas_size, head_num=Head_num, layer_num=TrLayer_num, embedding_dim=512,
                                   gru_gating=GRU_GATING, device=self.device)
        self.oass_en2 = nn.Linear(512, 256)

        self.obss_en1 = nn.Linear(obs_size, 32)

        self.layer1 = nn.Linear(256 + 32, 256)
        self.layer2 = nn.Linear(256, 256)
        self.flatten = torch.nn.Flatten()

        self.fc1 = nn.Linear(256, 256)
        self.logit_v = nn.Linear(256, 1)
        self.logit_p = nn.Linear(256, num_actions)

    def forward(self, x):
        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=True)

        N, B, _ = other_seq.shape
        self.transformer.set_v(B, N, self.device)

        key_padding_mask = []
        for i in range(B):
            for j in range(N):
                key_padding_mask.append(0 if j < num_other_agents[i] else 1)
        key_padding_mask = torch.from_numpy(np.array(key_padding_mask, dtype=np.float32)).view(B, N).to(
            device=self.device)

        oe2_v = oe2_s = torch.zeros(B, 256, device=self.device)
        obe_v = obe_s = torch.zeros(B, 32, device=self.device)
        ly1_v = ly1_s = torch.zeros(B, 256, device=self.device)
        ly2_v = ly2_s = torch.zeros(B, 256, device=self.device)
        fc1_v = fc1_s = torch.zeros(B, 256, device=self.device)
        lgv_v = lgv_s = torch.zeros(B, 1, device=self.device)
        lgp_v = lgp_s = torch.zeros(B, self.num_actions, device=self.device)

        for step in range(self.timesteps):
            trans_state = self.transformer(other_seq, attn_mask=key_padding_mask)['logit']
            a0 = torch.mean(trans_state, 0)
            oe2_v, oe2_s = self.mem_update(self.oass_en2, a0, oe2_v, oe2_s)
            obe_v, obe_s = self.mem_update(self.obss_en1, host_vec, obe_v, obe_s)

            cat_s = torch.cat((obe_s, oe2_s), dim=-1)

            ly1_v, ly1_s = self.mem_update(self.layer1, cat_s, ly1_v, ly1_s)
            ly2_v, ly2_s = self.mem_update(self.layer2, ly1_s, ly2_v, ly2_s)

            final_flat = self.flatten(ly2_s)

            fc1_v, fc1_s = self.mem_update(self.fc1, final_flat, fc1_v, fc1_s)

            lgv_v, lgv_s = self.ns_mem_update(self.logit_v, fc1_s, lgv_v, lgv_s)
            lgp_v, lgp_s = self.ns_mem_update(self.logit_p, fc1_s, lgp_v, lgp_s)

        value = torch.squeeze(lgv_v, 1)
        logit = lgp_v

        return logit, value

if __name__ == "__main__":

    model = SpikGTrNet().cuda()
    t = np.random.rand(1, 26+0*7)
    t[0][0]=2
    o=model(t)

    print(o)



