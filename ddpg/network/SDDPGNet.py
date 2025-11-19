import random
import numpy as np
import torch
import torch.nn as nn
import os

from ddpg.network.BaseNet import BaseNet

from layers.spikRNN import SpikingLSTM

NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

def state2ddpg_state(x,spike_flag=False):
    states = x[:, 1:]

    outs=[]
    for state in states:
        dis2goal=0.2/state[0]
        head_ego=state[1]/np.pi

        vx = state[2]/1.0
        vy = state[3]/1.0
        pref_speed = state[4]
        scans=0.2/state[6:]
        scans=np.clip(scans,0,1)

        if not spike_flag:
            out=[0 for _ in range(36+4)]
            out[0] = dis2goal
            out[1] = head_ego
            out[2] = vx
            out[3] = vy
            out[4:] = scans
        else:
            out = [0 for _ in range(36 + 6)]
            out[0] = dis2goal
            out[1] = head_ego if head_ego>0 else 0
            out[2] = abs(head_ego) if head_ego<0 else 0
            out[3] = vx
            out[4] = vy if vy>0 else 0
            out[5] = abs(vy) if vy < 0 else 0
            out[6:] = scans
        outs.append(out)

    # dis2goal=0.2/state[:,0]
    # head_ego=state[:,1]/np.pi
    # # speed_ego=state[:,2]
    # # pref_speed = state[:,3]
    # # scans=state[:,5:]/6.0
    #
    # vx = state[:, 2]/1.0
    # vy = state[:, 3]/1.0
    # pref_speed = state[:,4]
    # scans=state[:,6:]/6.0
    #
    # if not spike_flag:
    #     out=np.array([dis2goal,head_ego,vx,vy,scans])
    # else:
    #
    #     out=np.array([dis2goal,head_ego,vx,vy,scans])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outs = torch.Tensor(np.array(outs)).to(device)
    return outs

class SDDPG_ActorNet(BaseNet):
    """ Actor Network """
    def __init__(self, obs_size=4,oas_size=512,action_space=2,timesteps=20):
        super(SDDPG_ActorNet, self).__init__(obs_size,oas_size,action_space,timesteps)

        state_num = obs_size + oas_size + 2
        action_num = action_space
        batch_window = timesteps
        hidden1 = 256
        hidden2 = 256
        hidden3 = 256

        self.pseudo_spike = PseudoSpikeRect.apply

        self.state_num = state_num
        self.action_num = action_num
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.fc1 = nn.Linear(self.state_num, self.hidden1, bias=True)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=True)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3, bias=True)
        self.fc4 = nn.Linear(self.hidden3, self.action_num, bias=True)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, x):
        states = state2ddpg_state(x, spike_flag=True)

        batch_size, _ = states.shape

        fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_u = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_v = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_s = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc3_u = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_v = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_s = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc4_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_s = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)
        for step in range(self.batch_window):
            input = states>torch.rand(states.size(), device = self.device)
            input_spike = input.float().to(self.device)
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, input_spike, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc3_u, fc3_v, fc3_s = self.neuron_model(self.fc3, fc2_s, fc3_u, fc3_v, fc3_s)
            fc4_u, fc4_v, fc4_s = self.neuron_model(self.fc4, fc3_s, fc4_u, fc4_v, fc4_s)
            fc4_sumspike += fc4_s
        out = fc4_sumspike / self.batch_window

        return out

class DDPG_CriticNet(nn.Module):
    """ Critic Network"""
    def __init__(self, obs_size=4,oas_size=512, action_space=2):
        super(DDPG_CriticNet, self).__init__()

        state_num = obs_size + oas_size
        action_num= action_space
        hidden1 = 512
        hidden2 = 512
        hidden3 = 512

        self.fc1 = nn.Linear(state_num, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_num, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()

    def forward(self, xa):
        x, a = xa
        x = state2ddpg_state(x, spike_flag=False)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(torch.cat([x, a], 1)))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

    def reset_mean_std(self):
        return

if __name__ == "__main__":
    model=SDDPG_ActorNet().cuda()

    t = np.random.rand(1, 1+26+7*6)
    t[0][1]=8
    o = model(t)

    print(o)
