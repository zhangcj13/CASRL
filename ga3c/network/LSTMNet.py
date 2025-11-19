import os
import re
import numpy as np
# import tensorflow as tf
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import math
from ga3c.network.BaseNet import BaseNet


class LSTMNet(BaseNet):
    def __init__(self, obs_size=4, oas_size=7, num_actions=11):
        super(LSTMNet, self).__init__(obs_size, oas_size, num_actions)
        self.relu = nn.ReLU(inplace=False)
        self.lstm_dim = 64
        self.lstm_layer = 1

        self.lstm = nn.LSTM(oas_size, self.lstm_dim, self.lstm_layer, batch_first=False)

        self.layer1 = nn.Linear(obs_size + self.lstm_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.flatten = torch.nn.Flatten()

        self.fc1 = nn.Linear(256, 256)
        self.logit_v = nn.Linear(256, 1)
        self.logit_p = nn.Linear(256, num_actions)

    def forward(self, x):
        host_vec, other_seq, num_other_agents = self.rescale_input(x, seq_cut=True)
        N, B, _ = other_seq.shape

        h0 = torch.zeros(self.lstm_layer, B, self.lstm_dim).to(self.device)
        c0 = torch.zeros(self.lstm_layer, B, self.lstm_dim).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(other_seq, (h0, c0))

        # Dynamic RNN   Decode the hidden state of the last time step
        num_other_agents = np.clip(num_other_agents, 1, 1000)

        seq_length = torch.from_numpy(np.array(num_other_agents, dtype=np.int64)).to(self.device)
        last_step_index_list = (seq_length - 1).view(-1, 1).expand(out.size(1), out.size(2)).unsqueeze(0)
        rnn_output = out.gather(0, last_step_index_list).squeeze(0)

        a = torch.cat((host_vec, rnn_output), dim=-1)
        a = self.relu(self.layer1(a))
        a = self.relu(self.layer2(a))
        final_flat = self.flatten(a)
        fc = self.relu(self.fc1(final_flat))
        value = torch.squeeze(self.logit_v(fc), 1)
        logit = self.logit_p(fc)
        return logit, value
