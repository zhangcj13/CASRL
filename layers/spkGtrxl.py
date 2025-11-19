from typing import Optional, Dict, List
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ding.torch_utils.network.nn_module import fc_block, build_normalization, F
# import sys
# sys.path.append('D:/tools/_SelfWork/Robot/rl_collision_avoidance-release')
from layers.surrogate import pseudo_spike,erf,sigmoid

from layers.gtrxl import Memory,PositionalEmbedding


class SpikGRUGatingUnit(torch.nn.Module):
    def __init__(self, input_dim: int, bg: float = 2.,SpikNeg: bool = False):
        super(SpikGRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        
        self.surrogate_function = erf.apply
        self.SpikNeg=SpikNeg

        # self.sigmoid = torch.nn.Sigmoid()
        # self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        """
        r = self.surrogate_function(self.Wr(y) + self.Ur(x))
        z = self.surrogate_function(self.Wz(y) + self.Uz(x) - self.bg)
        if not self.SpikNeg:
            h = self.surrogate_function(self.Wg(y) + self.Ug(torch.mul(r, x)))
        else:
            raise NotImplemented
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        
        return g  # x.shape == y.shape == g.shape

class SpikAttentionXL(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        super(SpikAttentionXL, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = nn.Linear(input_dim, head_dim * head_num * 2)  # key, value
        self.attention_q = nn.Linear(input_dim, head_dim * head_num)  # query (not computed with past hidden states)
        self.project = nn.Linear(head_dim * head_num, input_dim)  # project attention output back to input_dim
        self.project_pos = nn.Linear(input_dim, head_dim * head_num)  # project the positional embedding
        self.scale = 1 / (head_dim ** 0.5)  # for scaled dot product attention

        self.surrogate_function=sigmoid.apply

        self.kv_v = self.kv_s = None
        self.q_v = self.q_s = None
        self.a_v = self.a_s = None
        self.r_v = self.r_s = None

    def _rel_shift(self, x: torch.Tensor, zero_upper: bool = False):
        x_padded = F.pad(x, [1, 0])  # step 1
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))  # step 2
        x = x_padded[:, :, 1:].view_as(x)  # step 3
        if zero_upper:
            ones = torch.ones((x.size(2), x.size(3))).unsqueeze(0).unsqueeze(0)
            x = x * torch.tril(ones.to(x.device), x.size(3) - x.size(2))  # step 4
        return x

    def charge_v(self, mem, volt, spike):
        volt = volt * (1. - spike) + mem
        spike = self.surrogate_function(volt)
        return volt, spike

    def set_v(self,B,N,device,ML):
        self.kv_v = self.kv_s=torch.zeros(N+ML,B, self.head_dim * self.head_num * 2, device=device)
        self.q_v = self.q_s =torch.zeros(N,B, self.head_dim * self.head_num, device=device)
        self.a_v = self.a_s =torch.zeros(B, self.head_num,N,N+ML, device=device)
        self.r_v = self.r_s =torch.zeros(N+ML,1, self.head_dim*self.head_num, device=device)
        return 
    def forward(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            full_input: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        self.kv_v, self.kv_s = self.charge_v(kv,self.kv_v, self.kv_s)
        key, value = torch.chunk(self.kv_s, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        self.q_v, self.q_s = self.charge_v(query, self.q_v, self.q_s)
        query=self.q_s
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head
        self.r_v, self.r_s = self.charge_v(r, self.r_v, self.r_s)
        r=self.r_s

        key = key.view(full_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.view(full_seq, self.head_num, self.head_dim)

        # (query + u) * key^T
        q_u = query + u
        content_attn = q_u.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)  # bs x head_num x cur_seq x full_seq

        # (query + v) * R^T
        q_v = query + v
        position_attn = q_v.permute(1, 2, 0, 3) @ r.permute(1, 2, 0)  # bs x head_num x cur_seq x full_seq
        position_attn = self._rel_shift(position_attn)

        attn = content_attn + position_attn  # bs x head_num x cur_seq x full_seq
        attn.mul_(self.scale)

        # fills float('-inf') where mask is True to let softmax ignore those positions.
        if mask is not None and mask.any().item():
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
            assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
            attn = attn.masked_fill(mask, -float("inf")).type_as(attn)

        # attn = F.softmax(attn, dim=-1)
        self.a_v,self.a_s=self.charge_v(attn, self.a_v, self.a_s)
        attn=self.a_s
        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


class SpikGatedTransformerXLLayer(torch.nn.Module):
    """
    Overview:
        Attention layer of GTrXL
    """

    def __init__(
            self,
            input_dim: int,
            head_dim: int,
            hidden_dim: int,
            head_num: int,
            mlp_num: int,
            dropout: nn.Module,
            gru_gating: bool = True,
            gru_bias: float = 2.,
            device='cpu'
    ) -> None:
        
        super(SpikGatedTransformerXLLayer, self).__init__()
        self.dropout = dropout
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = SpikGRUGatingUnit(input_dim, gru_bias)
            self.gate2 = SpikGRUGatingUnit(input_dim, gru_bias)
        self.attention = SpikAttentionXL(
            input_dim,
            head_dim,
            head_num,
            dropout,
        )
        
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        # layers = []
        # for i in range(mlp_num):
        #     layers.append(nn.Sequential(
        #         nn.Linear(dims[i], dims[i + 1]),
        #         activation)
        #     )
        #     if i != mlp_num - 1:
        #         layers.append(self.dropout)
        # layers.append(self.dropout)
        # self.mlp = nn.Sequential(*layers)
        self.dims=dims
        self.mlp_num = mlp_num
        self.mlp_layers=[]
        self.mlp_dropout=[]
        self.mlp_v=[]
        self.mlp_s=[]
        for i in range(mlp_num):
            self.mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.mlp_dropout.append(self.dropout)
            self.mlp_v.append(None)
            self.mlp_s.append(None)
            self.mlp_layers[i].to(device)

        # self.layernorm1 = nn.LayerNorm(input_dim)
        # self.layernorm2 = nn.LayerNorm(input_dim)

        self.surrogate_function=pseudo_spike.apply
        self.a1_v=self.a1_s=None

    def charge_v(self, mem, volt, spike):
        volt = volt * (1. - spike) + mem
        spike = self.surrogate_function(volt)
        return volt, spike

    def set_v(self,B,N,device,ML ):
        self.a1_v=self.a1_s=torch.zeros(N,B, self.dims[0], device=device)
        for i in range(self.mlp_num):
            self.mlp_v[i]=self.mlp_s[i]=torch.zeros(N,B, self.dims[i+1], device=device)

        self.attention.set_v(B,N,device,ML)
        return 

    def forward(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            memory: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # concat memory with input across sequence dimension
        full_input = torch.cat([memory, inputs], dim=0)  # full_seq x bs x input_dim
        # x1 = self.layernorm1(full_input)
        x1 = full_input
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        # a1 = self.activation(a1)  # RELU after attention
        self.a1_v,self.a1_s=self.charge_v(a1,self.a1_v,self.a1_s)
        o1 = self.gate1(inputs, self.a1_s) if self.gating else inputs + self.a1_s
        # x2 = self.layernorm2(o1)
        x2 = o1
        for i in range(self.mlp_num):
            x2=self.mlp_layers[i](x2)
            self.mlp_v[i],self.mlp_s[i]=self.charge_v(x2,self.mlp_v[i],self.mlp_s[i])
            x2=self.mlp_dropout[i](self.mlp_s[i])
        # m2 = self.dropout(self.mlp(x2))
        m2=self.dropout(x2)
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


class SpikGTrXL(nn.Module):
    def __init__(
            self,
            input_dim: int,
            head_dim: int = 128,
            embedding_dim: int = 256,
            head_num: int = 2,
            mlp_num: int = 2,
            layer_num: int = 3,
            memory_len: int = 64,
            dropout_ratio: float = 0.,
            gru_gating: bool = True,
            gru_bias: float = 2.,
            use_embedding_layer: bool = True,
            device='cpu'
    ) -> None:
        super(SpikGTrXL, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = nn.Sequential(nn.Linear(input_dim, embedding_dim))
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        # memory to save hidden states of past segments
        # it will be initialized in the forward method to get its size dynamically
        self.memory = None
        self.memory_len = memory_len
        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                SpikGatedTransformerXLLayer(
                    dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout, gru_gating,
                    gru_bias,device
                )
            )
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        # u and v are the parameters to compute global content bias and global positional bias
        self.u, self.v = (
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        )
        self.att_mask = {}  # create an attention mask for each different seq_len, in this way we don't need to create a
        # new one each time we call the forward method
        self.pos_embedding_dict = {}  # create a pos embedding for each different seq_len

        self.surrogate_function=pseudo_spike.apply
        self.ebd_v = self.ebd_s = None

    def reset_memory(self, batch_size: Optional[int] = None, state: Optional[torch.Tensor] = None):
        self.memory = Memory(memory_len=self.memory_len, layer_num=self.layer_num, embedding_dim=self.embedding_dim)
        if batch_size is not None:
            self.memory = Memory(self.memory_len, batch_size, self.embedding_dim, self.layer_num)
        elif state is not None:
            self.memory.init(state)

    def get_memory(self):
        if self.memory is None:
            return None
        else:
            return self.memory.get()

    def charge_v(self, mem, volt, spike):
        volt = volt * (1. - spike) + mem
        spike = self.surrogate_function(volt)
        return volt, spike

    def set_v(self, B,N, device):
        self.ebd_v = self.ebd_s = torch.zeros(N,B, self.embedding_dim, device=device)
        for i in range(self.layer_num):
            self.layers[i].set_v(B,N, device,self.memory_len)
        return 

    def forward(self, x: torch.Tensor, batch_first: bool = False, return_mem: bool = True,reset_mem: bool = False) -> Dict[str, torch.Tensor]:

        if batch_first:
            x = torch.transpose(x, 1, 0)  # bs x cur_seq x input_dim -> cur_seq x bs x input_dim
        cur_seq, bs = x.shape[:2]
        if self.ebd_v is None:
            self.set_v(bs,cur_seq,x.device)

        memory = None if self.memory is None else self.memory.get()
        if memory is None or reset_mem:
            self.reset_memory(bs)  # (layer_num+1) x memory_len x batch_size x embedding_dim
        elif memory.shape[-2] != bs or memory.shape[-1] != self.embedding_dim:
            warnings.warn(
                "Memory {} and Input {} dimensions don't match,"
                " this will cause the memory to be initialized to fit your input!".format(
                    list(memory.shape[-2:]), [x.shape[-2]] + [self.embedding_dim]
                )
            )
            self.reset_memory(bs)
        self.memory.to(x.device)
        memory = self.memory.get()

        if self.use_embedding_layer:
            x = self.embedding(x)
            self.ebd_v,self.ebd_s=self.charge_v(x,self.ebd_v,self.ebd_s)
            x = self.dropout(self.ebd_s)
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq

        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        else:
            attn_mask = (
                torch.triu(
                    torch.ones((cur_seq, full_seq)),
                    diagonal=1 + prev_seq,  # fixed in train, eval, collect
                ).bool().unsqueeze(-1).to(x.device)
            )  # cur_seq x full_seq x 1
            self.att_mask[cur_seq] = attn_mask

        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        else:
            pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)  # full_seq
            pos_embedding = self.pos_embedding(pos_ips.to(x.device))
            self.pos_embedding_dict[cur_seq] = pos_embedding
        pos_embedding = self.dropout(pos_embedding)  # full_seq x 1 x embedding_dim

        hidden_state = [x]
        out = x
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(
                out,
                pos_embedding,
                self.u,
                self.v,
                mask=attn_mask,
                memory=memory[i],  # (layer_num+1) x memory_len x batch_size x embedding_dim
            )  # cur_seq x bs x embedding_dim
            hidden_state.append(out.clone())

        out = self.dropout(out)
        self.memory.update(hidden_state)  # (layer_num+1) x memory_len x batch_size x embedding_dim

        if batch_first:
            out = torch.transpose(out, 1, 0)  # cur_seq x bs x embedding_dim -> bs x cur_seq x embedding_dim
        if return_mem:
            output = {"logit": out, "memory": memory}  # return the content of the memory before the last update
        else:
            output = {"logit": out}
        return output

if __name__ == "__main__":
    model=SpikGTrXL(input_dim=7, head_num=4, layer_num=1,embedding_dim=256)
    # network=GTrXL(input_dim=7, head_num=4, layer_num=1,embedding_dim=256)

    t = torch.rand(13,11,7)
    o = model(t)

    # o=GPTConfig()

    print(o)
