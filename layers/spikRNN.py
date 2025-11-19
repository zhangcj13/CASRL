import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.surrogate import erf

def bidirectional_rnn_cell_forward(cell: nn.Module, cell_reverse: nn.Module, x: torch.Tensor,
                                   states: torch.Tensor, states_reverse: torch.Tensor):

    T = x.shape[0]
    ss = states
    ss_r = states_reverse
    output = []
    output_r = []
    for t in range(T):
        ss = cell(x[t], ss)
        ss_r = cell_reverse(x[T - t - 1], ss_r)
        if states.dim() == 2:
            output.append(ss)
            output_r.append(ss_r)
        elif states.dim() == 3:
            output.append(ss[0])
            output_r.append(ss_r[0])
            # 当RNN cell具有多个隐藏状态时，通常第0个隐藏状态是其输出

    ret = []
    for t in range(T):
        ret.append(torch.cat((output[t], output_r[T - t - 1]), dim=-1))
    return torch.stack(ret), ss, ss_r


class SpikingRNNCellBase(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def reset_parameters(self):
        
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def weight_ih(self):
        
        return self.linear_ih.weight

    def weight_hh(self):
        
        return self.linear_hh.weight

    def bias_ih(self):
        
        return self.linear_ih.bias

    def bias_hh(self):
        
        return self.linear_hh.bias


class SpikingRNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False, *args, **kwargs):
        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout_p = dropout_p
        self.invariant_dropout_mask = invariant_dropout_mask
        self.bidirectional = bidirectional

        if self.bidirectional:
            # 双向LSTM的结构可以参考 https://cedar.buffalo.edu/~srihari/CSE676/10.3%20BidirectionalRNN.pdf
            # https://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
            self.cells, self.cells_reverse = self.create_cells(*args, **kwargs)

        else:
            self.cells = self.create_cells(*args, **kwargs)

    def create_cells(self, *args, **kwargs):
        
        if self.bidirectional:
            cells = []
            cells_reverse = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            cells_reverse.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
                cells_reverse.append(
                    self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells), nn.Sequential(*cells_reverse)

        else:
            cells = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells)

    @staticmethod
    def base_cell():
        
        raise NotImplementedError

    @staticmethod
    def states_num():
        
        # LSTM: 2
        # GRU: 1
        # RNN: 1
        raise NotImplementedError

    def forward(self, x: torch.Tensor, states=None):
        
        # x.shape=[T, batch_size, input_size]
        # states states_num 个 [num_layers * num_directions, batch, hidden_size]
        T = x.shape[0]
        batch_size = x.shape[1]

        if isinstance(states, tuple):
            # states非None且为tuple，则合并成tensor
            states_list = torch.stack(states)
            # shape = [self.states_num(), self.num_layers * 2, batch_size, self.hidden_size]
        elif isinstance(states, torch.Tensor):
            # states非None且不为tuple时，它本身就是一个tensor，例如普通RNN的状态
            states_list = states
        elif states is None:
            # squeeze(0)的作用是，若states_num() == 1则去掉多余的维度
            if self.bidirectional:
                states_list = torch.zeros(
                    size=[self.states_num(), self.num_layers * 2, batch_size, self.hidden_size]).to(x).squeeze(0)
            else:
                states_list = torch.zeros(size=[self.states_num(), self.num_layers, batch_size, self.hidden_size]).to(
                    x).squeeze(0)
        else:
            raise TypeError

        if self.bidirectional:
            # y 表示第i层的输出。初始化时，y即为输入
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size * 2]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)
            for i in range(self.num_layers):
                # 第i层神经元的起始状态从输入states_list获取
                new_states_list = torch.zeros_like(states_list.data)
                if self.states_num() == 1:
                    cell_init_states = states_list[i]
                    cell_init_states_reverse = states_list[i + self.num_layers]
                else:
                    cell_init_states = states_list[:, i]
                    cell_init_states_reverse = states_list[:, i + self.num_layers]

                if self.training and self.dropout_p > 0:
                    if i > 1:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                y, ss, ss_r = bidirectional_rnn_cell_forward(
                    self.cells[i], self.cells_reverse[i], y, cell_init_states, cell_init_states_reverse)
                # 更新states_list[i]
                if self.states_num() == 1:
                    new_states_list[i] = ss
                    new_states_list[i + self.num_layers] = ss_r
                else:
                    new_states_list[:, i] = torch.stack(ss)
                    new_states_list[:, i + self.num_layers] = torch.stack(ss_r)
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return y, new_states_list
            else:
                # split使得返回值是tuple
                return y, torch.split(new_states_list, 1, dim=0)

        else:
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)

            output = []

            for t in range(T):
                new_states_list = torch.zeros_like(states_list.data)
                if t==0:
                    self.cells[0].reset_v(x[t].shape,x[t].device)
                if self.states_num() == 1:
                    new_states_list[0] = self.cells[0](x[t], states_list[0])
                else:
                    new_states_list[:, 0] = torch.stack(self.cells[0](x[t], states_list[:, 0]))
                for i in range(1, self.num_layers):
                    y = states_list[0, i - 1]
                    if self.training and self.dropout_p > 0:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                    if self.states_num() == 1:
                        new_states_list[i] = self.cells[i](y, states_list[i])
                    else:
                        new_states_list[:, i] = torch.stack(self.cells[i](y, states_list[:, i]))
                if self.states_num() == 1:
                    output.append(new_states_list[-1].clone().unsqueeze(0))
                else:
                    output.append(new_states_list[0, -1].clone().unsqueeze(0))
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return torch.cat(output, dim=0), new_states_list
            else:
                # split使得返回值是tuple
                return torch.cat(output, dim=0), torch.split(new_states_list, 1, dim=0)


class SpikingLSTMCell(SpikingRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        
        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.Spike_Neg=False

        # self.spikefunc = PseudoSpikeRect.apply
        self.spikefunc = erf.apply
        self.reset_parameters()
        self.u = self.v = self.s = None

        self.i_u = self.i_v = self.i_s =\
            self.f_u = self.f_v = self.f_s =\
            self.o_u = self.o_v = self.o_s =\
            self.g_u = self.g_v = self.g_ps = self.g_ns = None

    def reset_v(self,shape,device=None):
        B, N = shape
        # self.u = self.v= self.s = torch.zeros(B, self.hidden_size*4, device=device)
        self.i_u = self.i_v = self.i_s = torch.zeros(B, self.hidden_size, device=device)
        self.f_u = self.f_v = self.f_s = torch.zeros(B, self.hidden_size, device=device)
        self.o_u = self.o_v = self.o_s = torch.zeros(B, self.hidden_size, device=device)
        self.g_u = self.g_v = self.g_ps = self.g_ns = torch.zeros(B, self.hidden_size, device=device)

    def charge_v(self, mem, current, volt, spike):
        current = current * 0.1 + mem
        volt = volt * 1.0 * (1. - spike) + current
        spike = self.spikefunc(volt)

        return current, volt, spike
    def charge_npv(self, mem, current, volt, pspike,nspike):
        current = current * 0.1 + mem
        # print(current,mem)
        volt = volt * 1.0 * (1. - pspike)*(1. - nspike) + current
        pspike = self.spikefunc(volt)
        nspike = self.spikefunc(-volt)
        return current, volt, pspike, nspike

    def forward(self, x: torch.Tensor, hc=None):
        
        if hc is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
            c = torch.zeros_like(h)
        else:
            h = hc[0]
            c = hc[1]

        if self.Spike_Neg:
            i, f, g, o = torch.split(self.linear_ih(x) + self.linear_hh(h), self.hidden_size, dim=1)

            self.i_u, self.i_v, self.i_s = self.charge_v(i, self.i_u, self.i_v, self.i_s)
            self.f_u, self.f_v, self.f_s = self.charge_v(f, self.f_u, self.f_v, self.f_s)
            self.o_u, self.o_v, self.o_s = self.charge_v(o, self.o_u, self.o_v, self.o_s)
            self.g_u, self.g_v, self.g_ps, self.g_ns = self.charge_npv(g, self.g_u, self.g_v, self.g_ps, self.g_ns)
            i = self.i_s
            f = self.f_s
            o = self.o_s
            g = self.g_ps - self.g_ns
            '''
            i = self.surrogate_function1(i)
            f = self.surrogate_function1(f)
            g = self.surrogate_function2(g)
            o = self.surrogate_function1(o)
            '''
            # membrane = self.linear_ih(x) + self.linear_hh(h)
            # self.u, self.v, self.s=self.charge_v(membrane,self.u, self.v, self.s)
            # i, f, g, o = torch.split(self.s,self.hidden_size, dim=1)

            c = c * f + i * g
            h = c * o
        else:
            i, f, g, o = torch.split(self.linear_ih(x) + self.linear_hh(h), self.hidden_size, dim=1)
            i = self.spikefunc(i)
            f = self.spikefunc(f)
            g = self.spikefunc(g)
            c = c * f + i * g
            o = self.spikefunc(o)
            h = c * o
        return h, c


class SpikingLSTM(SpikingRNNBase):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False):
        
        super().__init__(input_size, hidden_size, num_layers, bias, dropout_p, invariant_dropout_mask, bidirectional)

    @staticmethod
    def base_cell():
        return SpikingLSTMCell

    @staticmethod
    def states_num():
        return 2


