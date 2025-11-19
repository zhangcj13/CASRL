import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from dppo.network.BaseNet import BaseNet
from layers.spikGTrAN import SpikGTrAN

class SpikGTrANActor(BaseNet):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None,continuous_action_space=True,timestep=3):
        super(SpikGTrANActor, self).__init__(obs_size,oas_size, action_space,device,timestep)

        self.act_fea_transformer = SpikGTrAN(input_dim=oas_size, head_num=8, layer_num=1,embedding_dim=512,gru_gating=True,device=self.device)

        self.act_fc1 = nn.Linear(512, 512)

        self.act_fea_obs = nn.Linear(obs_size, 32)

        self.act_fc2 = nn.Linear(512 + 32, 256)
        self.act = nn.Linear(256, action_space)

        self.continuous_action_space=continuous_action_space

    def forward(self, x):

        host_vec, other_seq, num_other_agents = x

        N, B, _ = other_seq.shape
        self.act_fea_transformer.set_v(B, N, other_seq.device)

        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length).view(-1, 1).expand(B, 512).unsqueeze(0)

        fc1_v = fc1_s = torch.zeros(B, 512, device=self.device)
        obe_v = obe_s = torch.zeros(B, 32, device=self.device)
        fc2_v = fc2_s = torch.zeros(B, 256, device=self.device)
        act_v = act_s = torch.zeros(B, self.action_space, device=self.device)

        for step in range(self.timesteps):
            trans_state = self.act_fea_transformer(other_seq)['logit']
            # a_tran = torch.mean(trans_state, 0)
            # fc1_v, fc1_s = self.mem_update(self.act_fc1, a_tran, fc1_v, fc1_s)

            tran_mems = torch.zeros(N + 1, B, 512, device=self.device)
            for sl in range(N):
                cur_mem = self.act_fc1(trans_state[sl])
                tran_mems[sl + 1] = cur_mem + tran_mems[sl]
            tran_mem = tran_mems.gather(0, last_step_index_list).squeeze(0)

            fc1_v, fc1_s = self.charge_v(tran_mem, fc1_v, fc1_s)
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

    def get_spike(self,x):
        host_vec, other_seq, num_other_agents = x

        N, B, _ = other_seq.shape
        self.act_fea_transformer.set_v(B, N, other_seq.device)

        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length).view(-1, 1).expand(B, 512).unsqueeze(0)

        fc1_v = fc1_s = torch.zeros(B, 512, device=self.device)
        obe_v = obe_s = torch.zeros(B, 32, device=self.device)
        fc2_v = fc2_s = torch.zeros(B, 256, device=self.device)
        act_v = act_s = torch.zeros(B, self.action_space, device=self.device)

        # output_spks = torch.zeros(self.timesteps,B, 512, device=self.device)
        output_spks = torch.zeros( B, 512, device=self.device)

        sum_spk_embd = 0
        sum_spk_kv = 0
        sum_spk_q = 0
        sum_spk_attn = 0
        sum_spk_pro=0

        sum_spk_g1r = 0
        sum_spk_g1z = 0
        sum_spk_g1h = 0

        sum_spk_g2r = 0
        sum_spk_g2z = 0
        sum_spk_g2h = 0

        sum_spk_mlp1 = 0
        sum_spk_mlp2 = 0

        sum_spk_fc1 = 0
        sum_spk_obe = 0
        sum_spk_fc2 = 0

        for step in range(self.timesteps):
            trans_state = self.act_fea_transformer(other_seq)['logit']
            tran_mems = torch.zeros(N + 1, B, 512, device=self.device)
            for sl in range(N):
                cur_mem = self.act_fc1(trans_state[sl])
                tran_mems[sl + 1] = cur_mem + tran_mems[sl]
            tran_mem = tran_mems.gather(0, last_step_index_list).squeeze(0)

            fc1_v, fc1_s = self.charge_v(tran_mem, fc1_v, fc1_s)
            obe_v, obe_s = self.mem_update(self.act_fea_obs, host_vec, obe_v, obe_s)

            # output_spks[step] = fc1_s
            output_spks += fc1_s

            cat_s = torch.cat((obe_s, fc1_s), dim=-1)

            fc2_v, fc2_s = self.mem_update(self.act_fc2, cat_s, fc2_v, fc2_s)
            act_v, act_s = self.ns_mem_update(self.act, fc2_s, act_v, act_s)

            sum_spk_embd += torch.sum(self.act_fea_transformer.ebd_s).cpu().detach().numpy()
            sum_spk_kv += torch.sum(self.act_fea_transformer.layers[0].attention.kv_s).cpu().detach().numpy()
            sum_spk_q += torch.sum(self.act_fea_transformer.layers[0].attention.q_s).cpu().detach().numpy()
            sum_spk_attn += torch.sum(self.act_fea_transformer.layers[0].attention.a_s).cpu().detach().numpy()
            sum_spk_pro += torch.sum(self.act_fea_transformer.layers[0].a1_s).cpu().detach().numpy()

            sum_spk_g1r += torch.sum(self.act_fea_transformer.layers[0].gate1.r).cpu().detach().numpy()
            sum_spk_g1z += torch.sum(self.act_fea_transformer.layers[0].gate1.z).cpu().detach().numpy()
            sum_spk_g1h += torch.sum(self.act_fea_transformer.layers[0].gate1.h).cpu().detach().numpy()

            sum_spk_g2r += torch.sum(self.act_fea_transformer.layers[0].gate2.r).cpu().detach().numpy()
            sum_spk_g2z += torch.sum(self.act_fea_transformer.layers[0].gate2.z).cpu().detach().numpy()
            sum_spk_g2h += torch.sum(self.act_fea_transformer.layers[0].gate2.h).cpu().detach().numpy()

            sum_spk_mlp1 += torch.sum(self.act_fea_transformer.layers[0].mlp_s[0]).cpu().detach().numpy()
            sum_spk_mlp2 += torch.sum(self.act_fea_transformer.layers[0].mlp_s[1]).cpu().detach().numpy()

            sum_spk_fc1 += torch.sum(fc1_s).cpu().detach().numpy()
            sum_spk_obe += torch.sum(obe_s).cpu().detach().numpy()
            sum_spk_fc2 += torch.sum(fc2_s).cpu().detach().numpy()

        print(sum_spk_embd,sum_spk_kv,sum_spk_q,sum_spk_attn,sum_spk_pro,sum_spk_g1r,sum_spk_g1z,sum_spk_g1h, sum_spk_mlp1,sum_spk_mlp2, sum_spk_g2r,sum_spk_g2z,sum_spk_g2h,sum_spk_fc1,sum_spk_obe,sum_spk_fc2)

        # return output_spks

        # value = act_v
        if self.continuous_action_space:
            output = F.tanh(act_v)
        else:
            output = F.softmax(act_v, dim=1)
        return output