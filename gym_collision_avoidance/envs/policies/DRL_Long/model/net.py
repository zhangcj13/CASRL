import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from ..model.utils import log_normal_density
# from network.utils import log_normal_density


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.shape[0], 1,  -1)

class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space):
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)


        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)



    def forward(self, x, goal, speed):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)


        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        v, _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(MLPPolicy, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # value network
        self.value_fc1 = nn.Linear(obs_space, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

class OASS_LSTMPolicy(nn.Module):
    def __init__(self, obs_size=4,oas_size=6, action_space=2):
        super(OASS_LSTMPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.lstm_dim = 256
        self.lstm_layer = 1

        self.act_oass_lstm = nn.LSTM(oas_size, self.lstm_dim, self.lstm_layer, batch_first=False)
        self.act_fc1 = nn.Linear(self.lstm_dim, 256)
        self.act_fc2 =  nn.Linear(obs_size+256, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

        self.crt_oass_lstm = nn.LSTM(oas_size, self.lstm_dim, self.lstm_layer, batch_first=False)
        self.crt_fc1 = nn.Linear(self.lstm_dim, 256)
        self.crt_fc2 = nn.Linear(obs_size+256, 128)

        self.critic = nn.Linear(128, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs, oass, batch_size=None):
        """
            returns value estimation, action, log_action_prob
        """
        # print('forward oass_tensor size: ',oass.size(),'\n',oass)
        if batch_size is None:
            batch_size=obs.size(0)

        oass=torch.transpose(oass,0,1)

        act_h = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        act_c = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        act_out, _ = self.act_oass_lstm(oass, (act_h, act_c))
        lst_act_h=act_out[-1, :, :]
        act_lstm = F.relu(self.act_fc1(lst_act_h))
        # action
        a = torch.cat((obs,act_lstm), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)


        crt_h = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        crt_c = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        crt_out, _ = self.crt_oass_lstm(oass, (crt_h, crt_c))
        lst_crt_h=crt_out[-1, :, :]
        crt_lstm = F.relu(self.crt_fc1(lst_crt_h))
        # value
        v = torch.cat((obs,crt_lstm), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, obs, oass, action,batch_size=None):
        v, _, _, mean = self.forward(obs,oass,batch_size)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

class OASS_ConvLSTMPolicy(nn.Module):
    def __init__(self, obs_size=4,oas_size=6, action_space=2):
        super(OASS_ConvLSTMPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.lstm_dim = 256
        self.lstm_layer = 1

        self.act_oass_lstm = nn.LSTM(oas_size, self.lstm_dim, self.lstm_layer, batch_first=False)
        self.act_fc1 = nn.Linear(self.lstm_dim, 256)
        self.act_fc2 =  nn.Linear(obs_size+256, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

        self.crt_oass_lstm = nn.LSTM(oas_size, self.lstm_dim, self.lstm_layer, batch_first=False)
        self.crt_fc1 = nn.Linear(self.lstm_dim, 256)
        self.crt_fc2 = nn.Linear(obs_size+256, 128)

        self.critic = nn.Linear(128, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs, oass, batch_size=None):
        """
            returns value estimation, action, log_action_prob
        """
        # print('forward oass_tensor size: ',oass.size(),'\n',oass)
        if batch_size is None:
            batch_size=obs.size(0)

        oass=torch.transpose(oass,0,1)

        act_h = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        act_c = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        act_out, _ = self.act_oass_lstm(oass, (act_h, act_c))
        lst_act_h=act_out[-1, :, :]
        act_lstm = F.relu(self.act_fc1(lst_act_h))
        # action
        a = torch.cat((obs,act_lstm), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)


        crt_h = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        crt_c = torch.zeros(self.lstm_layer, batch_size, self.lstm_dim).to(self.device)
        crt_out, _ = self.crt_oass_lstm(oass, (crt_h, crt_c))
        lst_crt_h=crt_out[-1, :, :]
        crt_lstm = F.relu(self.crt_fc1(lst_crt_h))
        # value
        v = torch.cat((obs,crt_lstm), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, obs, oass, action,batch_size=None):
        v, _, _, mean = self.forward(obs,oass,batch_size)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

class SOASS_CNNPolicy(nn.Module):
    def __init__(self, length=167,frames=3, action_space=2):
        super(SOASS_CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        neurons=256

        # self.act_oass = nn.Linear(6, neurons)
        # self.act_fc1 = nn.Linear(4+neurons, neurons)

        self.act_fc1 = nn.Linear(length, neurons)
        self.act_fc2 =  nn.Linear(neurons, neurons)
        self.actor1 = nn.Linear(neurons, 1)
        self.actor2 = nn.Linear(neurons, 1)

        # self.crt_oass = nn.Linear(6, neurons)
        # self.crt_fc1 = nn.Linear(4 + neurons, neurons)
        self.crt_fc1 = nn.Linear(length, neurons)
        self.crt_fc2 = nn.Linear(neurons, neurons)
        self.critic = nn.Linear(neurons, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs, batch_size=None):
        """
            returns value estimation, action, log_action_prob
        """
        # print('forward oass_tensor size: ',oass.size(),'\n',oass)
        if batch_size==None:
            batch_size=obs.size(0)
        # action

        # obss = obs[:,0:4]
        # oass = obs[:, 4:10]
        # a0 = F.relu(self.act_oass(oass))
        # a = torch.cat((obss, a0), dim=-1)
        # a = F.relu(self.act_fc1(a))

        a = F.relu(self.act_fc1(obs))
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        # v0 = F.relu(self.crt_oass(oass))
        # v = torch.cat((obss, v0), dim=-1)
        # v = F.relu(self.act_fc1(v))

        v = F.relu(self.crt_fc1(obs))
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, obs, action,batch_size=None):
        v, _, _, mean = self.forward(obs,batch_size)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

class SCAN_CNNPolicy(nn.Module):
    def __init__(self, obs_size=4,scan_size=360, action_space=2):
        super(SCAN_CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(scan_size*32//4, 256)
        self.act_fc2 =  nn.Linear(256+obs_size, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)


        self.crt_fea_cv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(scan_size*32//4, 256)
        self.crt_fc2 = nn.Linear(256+obs_size, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs, scan, batch_size=None):
        """
            returns value estimation, action, log_action_prob
        """
        if batch_size is None:
            batch_size=obs.size()[0]

        # action
        # print(obs.size(),scan.size())
        a = F.relu(self.act_fea_cv1(scan))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))
        # zeros=torch.zeros((batch_size,256)).cuda()
        # a = torch.cat((zeros, obs), dim=-1)
        a = torch.cat((a, obs), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        v = F.relu(self.crt_fea_cv1(scan))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, obs), dim=-1)
        # v = torch.cat((zeros, obs), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, obs,scan, action,batch_size=None):
        v, _, _, mean = self.forward(obs, scan,batch_size=None)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


if __name__ == '__main__':
    from torch.autograd import Variable

    net = MLPPolicy(3, 2)

    observation = Variable(torch.randn(2, 3))
    v, action, logprob, mean = net.forward(observation)
    print(v)

