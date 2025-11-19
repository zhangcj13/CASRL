from typing import Optional, Dict, List
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.surrogate import pseudo_spike,erf,sigmoid

class SpikGRUGatingUnit(torch.nn.Module):
    def __init__(self, input_dim: int, bg: float = 2., SpikNeg: bool = False):
        super(SpikGRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias

        self.surrogate_function = erf.apply
        self.SpikNeg = SpikNeg

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

        self.r = r
        self.z = z
        self.h = h

        return g  # x.shape == y.shape == g.shape


class SpikAttention(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        super(SpikAttention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = nn.Linear(input_dim, head_dim * head_num * 2)  # key, value
        self.attention_q = nn.Linear(input_dim, head_dim * head_num)  # query (not computed with past hidden states)
        self.project = nn.Linear(head_dim * head_num, input_dim)  # project attention output back to input_dim
        self.scale = 1 / (head_dim ** 0.5)  # for scaled dot product attention

        self.surrogate_function = sigmoid.apply

        self.kv_v = self.kv_s = None
        self.q_v = self.q_s = None
        self.a_v = self.a_s = None

        self.norm_attn_flag=True

    def charge_v(self, mem, volt, spike):
        volt = volt * (1. - spike) + mem
        spike = self.surrogate_function(volt)
        return volt, spike

    def set_v(self, B, N, device):
        self.kv_v = self.kv_s = torch.zeros(N, B, self.head_dim * self.head_num * 2, device=device)
        self.q_v = self.q_s = torch.zeros(N, B, self.head_dim * self.head_num, device=device)
        self.a_v = self.a_s = torch.zeros(B, self.head_num, N, N, device=device)
        return

    def forward(
            self,
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cur_seq, bs, _ = inputs.size()

        kv = self.attention_kv(inputs)
        self.kv_v, self.kv_s = self.charge_v(kv, self.kv_v, self.kv_s)
        key, value = torch.chunk(self.kv_s, 2, dim=-1)  # full_seq x bs x num_head*dim_head

        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        self.q_v, self.q_s = self.charge_v(query, self.q_v, self.q_s)
        query = self.q_s

        key = key.view(cur_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq, bs, self.head_num, self.head_dim)

        if mask is not None:# and mask.any().item():
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
        if key_padding_mask is not None and key_padding_mask.any().item():
            assert key_padding_mask.shape == (bs, cur_seq), \
                f"expecting key_padding_mask shape of {(bs, cur_seq)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bs, 1, 1, cur_seq). \
                expand(-1, self.head_num, cur_seq, -1).reshape(bs, self.head_num, cur_seq, cur_seq).bool()
            if mask is None:
                mask = key_padding_mask
            elif mask.dtype == torch.bool:
                mask = mask.logical_or(key_padding_mask)
            else:
                mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
            # key_paddin
        attn = (query.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)) * self.scale  # scaled dot product
        if mask is not None:
            assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
            attn = attn.masked_fill(mask, -float("inf")).type_as(attn)

        self.a_v, self.a_s = self.charge_v(attn, self.a_v, self.a_s)
        attn = self.a_s
        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)

        if self.norm_attn_flag:
            norm_vec = torch.from_numpy(np.array([1 / (n + 1) for n in range(cur_seq)], dtype=np.float32)).to(
                inputs.device)
            norm_vec = norm_vec.view(cur_seq, 1,1)
            attn_vec = attn_vec*norm_vec

        # cur_seq x bs x head_num * head_dim
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


class SpikGatedTransformerLayer(torch.nn.Module):

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

        super(SpikGatedTransformerLayer, self).__init__()
        self.dropout = dropout
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = SpikGRUGatingUnit(input_dim, gru_bias)
            self.gate2 = SpikGRUGatingUnit(input_dim, gru_bias)
        self.attention = SpikAttention(
            input_dim,
            head_dim,
            head_num,
            dropout,
        )

        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]

        self.dims = dims
        self.mlp_num = mlp_num
        self.mlp_layers = []
        self.mlp_dropout = []
        self.mlp_v = []
        self.mlp_s = []
        for i in range(mlp_num):
            self.mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.mlp_dropout.append(self.dropout)
            self.mlp_v.append(None)
            self.mlp_s.append(None)
            self.mlp_layers[i].to(device)

        self.surrogate_function = pseudo_spike.apply
        self.a1_v = self.a1_s = None

    def charge_v(self, mem, volt, spike):
        volt = volt * (1. - spike) + mem
        spike = self.surrogate_function(volt)
        return volt, spike

    def set_v(self, B, N, device):
        self.a1_v = self.a1_s = torch.zeros(N, B, self.dims[0], device=device)
        for i in range(self.mlp_num):
            self.mlp_v[i] = self.mlp_s[i] = torch.zeros(N, B, self.dims[i + 1], device=device)

        self.attention.set_v(B, N, device)
        return

    def forward(
            self,
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        a1 = self.dropout(self.attention(inputs, mask=mask,key_padding_mask=key_padding_mask))
        self.a1_v, self.a1_s = self.charge_v(a1, self.a1_v, self.a1_s)
        o1 = self.gate1(inputs, self.a1_s) if self.gating else inputs + self.a1_s
        x2 = o1
        for i in range(self.mlp_num):
            x2 = self.mlp_layers[i](x2)
            self.mlp_v[i], self.mlp_s[i] = self.charge_v(x2, self.mlp_v[i], self.mlp_s[i])
            x2 = self.mlp_dropout[i](self.mlp_s[i])
        m2 = self.dropout(x2)
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


class SpikGTrAN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            head_dim: int = 128,
            embedding_dim: int = 256,
            head_num: int = 2,
            mlp_num: int = 2,
            layer_num: int = 3,
            dropout_ratio: float = 0.,
            gru_gating: bool = True,
            gru_bias: float = 2.,
            use_embedding_layer: bool = True,
            device='cpu'
    ) -> None:
        super(SpikGTrAN, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = nn.Sequential(nn.Linear(input_dim, embedding_dim))

        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                SpikGatedTransformerLayer(
                    dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout, gru_gating,
                    gru_bias, device
                )
            )
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

        self.surrogate_function = pseudo_spike.apply
        self.ebd_v = self.ebd_s = None

    def charge_v(self, mem, volt, spike):
        volt = volt * (1. - spike) + mem
        spike = self.surrogate_function(volt)
        return volt, spike

    def set_v(self, B, N, device):
        self.ebd_v = self.ebd_s = torch.zeros(N, B, self.embedding_dim, device=device)
        for i in range(self.layer_num):
            self.layers[i].set_v(B, N, device)
        return

    def forward(self, x: torch.Tensor, batch_first: bool = False, attn_mask: torch.Tensor = None,key_padding_mask: torch.Tensor=None) -> Dict[
        str, torch.Tensor]:

        if batch_first:
            x = torch.transpose(x, 1, 0)  # bs x cur_seq x input_dim -> cur_seq x bs x input_dim
        cur_seq, bs = x.shape[:2]
        if self.ebd_v is None:
            self.set_v(bs, cur_seq, x.device)

        if self.use_embedding_layer:
            x = self.embedding(x)
            self.ebd_v, self.ebd_s = self.charge_v(x, self.ebd_v, self.ebd_s)
            x = self.dropout(self.ebd_s)

        if attn_mask is None:
            attn_mask = (
                torch.triu(
                    torch.ones((cur_seq, cur_seq)),
                    diagonal=1,  # fixed in train, eval, collect
                ).bool().unsqueeze(-1).to(x.device)
            )  # cur_seq x full_seq x 1

        out = x
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out, mask=attn_mask,key_padding_mask=key_padding_mask)  # cur_seq x bs x embedding_dim

        out = self.dropout(out)

        if batch_first:
            out = torch.transpose(out, 1, 0)  # cur_seq x bs x embedding_dim -> bs x cur_seq x embedding_dim

        output = {"logit": out}
        return output


if __name__ == "__main__":
    model = SpikGTrAN(input_dim=7, head_num=4, layer_num=1, embedding_dim=256)
    t = torch.rand(3, 1, 7)

    attn_mask = torch.randint(0, 1, (1, 3))

    out = model(t, key_padding_mask=attn_mask)

    print(out["logit"].size())
