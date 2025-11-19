import random
import numpy as np
import torch
import torch.nn as nn
import os

from ddpg.network.BaseNet import BaseNet

class LSTM_ActorNet(BaseNet):
    """ Actor Network """

    def __init__(self, obs_size=4, oas_size=7, action_space=2):
        super(LSTM_ActorNet, self).__init__(obs_size, oas_size, action_space)

        self.lstm_dim = 256
        self.lstm_layer = 1
        self.act_oass_lstm = nn.LSTM(oas_size, self.lstm_dim, self.lstm_layer, batch_first=False)
        self.fc1 = nn.Linear(self.lstm_dim, 256)

        self.fc2 = nn.Linear(256 + obs_size, 256)
        self.fc3 = nn.Linear(256, 256)

        self.actor1 = nn.Linear(256, 1)
        self.actor2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=True)

        N, B, _ = other_seq.shape

        act_h = torch.zeros(self.lstm_layer, B, self.lstm_dim).to(self.device)
        act_c = torch.zeros(self.lstm_layer, B, self.lstm_dim).to(self.device)

        out, _ = self.act_oass_lstm(other_seq, (act_h, act_c))

        num_other_agents = np.clip(num_other_agents, 1, 1000)
        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(out.size(1), out.size(2)).unsqueeze(0)
        lst_act_h = out.gather(0, last_step_index_list).squeeze(0)
        # lst_act_h = act_out[-1, :, :]
        a0 = self.relu(self.fc1(lst_act_h))
        a = torch.cat((host_vec, a0), dim=-1)
        a = self.relu(self.fc2(a))
        a = self.relu(self.fc3(a))

        a1 = self.sigmoid(self.actor1(a))
        a2 = self.tanh(self.actor2(a))
        out = torch.cat((a1, a2), dim=-1)
        return out

class LSTM_CriticNet(BaseNet):
    """ Critic Network"""
    def __init__(self, obs_size=4,oas_size=2,action_space=2):
        super(LSTM_CriticNet, self).__init__(obs_size,oas_size,action_space)
        self.lstm_dim = 512
        self.lstm_layer = 1
        self.crt_oass_lstm = nn.LSTM(oas_size, self.lstm_dim, self.lstm_layer, batch_first=False)

        self.fc1 = nn.Linear(self.lstm_dim, 512)
        self.fc2 = nn.Linear(512 + obs_size + action_space, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, xa):
        x, a = xa

        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=True)
        N, B, _ = other_seq.shape

        crt_h = torch.zeros(self.lstm_layer, B, self.lstm_dim).to(self.device)
        crt_c = torch.zeros(self.lstm_layer, B, self.lstm_dim).to(self.device)

        out, _ = self.crt_oass_lstm(other_seq, (crt_h, crt_c))

        num_other_agents = np.clip(num_other_agents, 1, 1000)
        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length-1).view(-1, 1).expand(out.size(1), out.size(2)).unsqueeze(0)
        lst_crt_h = out.gather(0, last_step_index_list).squeeze(0)

        v0 = self.relu(self.fc1(lst_crt_h))

        v = torch.cat((host_vec, v0, a), dim=-1)

        v = self.relu(self.fc2(v))
        v = self.relu(self.fc3(v))
        out = self.fc4(v)
        return out


if __name__ == "__main__":
    model=LSTM_ActorNet().cuda()

    t = np.random.rand(1, 26+7*20)
    t[0][0]=8
    o = model(t)

    # o=GPTConfig()

    print(o)
