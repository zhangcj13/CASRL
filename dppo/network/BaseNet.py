import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.nn import functional as F

from layers.surrogate import pseudo_spike,DECAY

class BaseNet(nn.Module):
    def __init__(self, obs_size=4,oas_size=7, action_space=2,device=None,timestep=5):
        super(BaseNet, self).__init__()
        self.device=device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_space = action_space
        self.logstd = nn.Parameter(torch.zeros(action_space))
        self.spikefunc = pseudo_spike.apply
        self.timesteps = timestep
        self.decay = DECAY

    def mem_update(self,ops, x, mem, spike):
        mem = mem * self.decay * (1. - spike) + ops(x)
        spike = self.spikefunc(mem)  # act_fun : approximation firing function
        return mem, spike

    def ns_mem_update(self,ops, x, mem, spike):
        mem = mem * self.decay + ops(x)
        spike = self.spikefunc(mem)
        return mem, spike

    def charge_v(self, current, mem, spike):
        mem = mem * self.decay * (1. - spike) + current
        spike = self.spikefunc(mem)  # act_fun : approximation firing function
        return mem, spike

    def forward(self, x):
        raise NotImplementedError






