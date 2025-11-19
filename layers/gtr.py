from typing import Optional, Dict, List
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.gtrxl import GRUGatingUnit

def get_key_padding_mask(batch_size,seq_length,padding,device=None):
    key_padding_mask = []
    for i in range(batch_size):
        for j in range(seq_length):
            key_padding_mask.append(0 if j < padding[i] else 1)
    key_padding_mask = torch.from_numpy(np.array(key_padding_mask, dtype=np.float32)).view(batch_size, seq_length)
    if device is not None:
        key_padding_mask=key_padding_mask.to(device=device)
    return key_padding_mask

def get_triu_attn_mask(seq_length, device=None):
    attn_mask = (
        torch.triu(
            torch.ones((seq_length, seq_length)),
            diagonal=1,  # fixed in train, eval, collect
        ).bool().unsqueeze(-1)#.to(x.device)
    )  # cur_seq x full_seq x 1
    if device is not None:
        attn_mask=attn_mask.to(device=device)
    return attn_mask


class Attention(torch.nn.Module):

    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:

        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = nn.Linear(input_dim, head_dim * head_num * 2)  # key, value
        self.attention_q = nn.Linear(input_dim, head_dim * head_num)  # query (not computed with past hidden states)
        self.project = nn.Linear(head_dim * head_num, input_dim)  # project attention output back to input_dim
        self.scale = 1 / (head_dim ** 0.5)  # for scaled dot product attention

        self.RFL=False# True
        if self.RFL:
            self.norm =  nn.LayerNorm(head_dim * head_num)

    def forward(
            self,
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        cur_seq, bs, _= inputs.size()

        kv = self.attention_kv(inputs)
        key, value = torch.chunk(kv, 2, dim=-1)
        query = self.attention_q(inputs)

        key = key.view(cur_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq, bs, self.head_num, self.head_dim)

        if mask is not None:# and mask.any().item():
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
        if key_padding_mask is not None and key_padding_mask.any().item():
            assert key_padding_mask.shape == (bs, cur_seq), \
                f"expecting key_padding_mask shape of {(bs, cur_seq)}, but got {key_padding_mask.shape}"
            # key_padding_mask=mask
            key_padding_mask = key_padding_mask.view(bs, 1, 1, cur_seq). \
                expand(-1, self.head_num, cur_seq, -1).reshape(bs , self.head_num, cur_seq, cur_seq).bool()

            # key_padding_mask = key_padding_mask.view(bs, 1, 1, cur_seq). \
            #     expand(-1, self.head_num, -1, -1).reshape(bs * self.head_num, 1, cur_seq)*(-1e6)

            if mask is None:
                mask = key_padding_mask
            elif mask.dtype == torch.bool:
                mask = mask.logical_or(key_padding_mask)
            else:
                mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # tq=query.permute(1, 2, 0, 3).flatten(0, 1)
        # tk=key.permute(1, 2, 3, 0).flatten(0, 1)
        # attn_output_weights = torch.baddbmm(key_padding_mask, tq,tk )
        # key_t = key.transpose(-2, -1)  # transpose
        attn = (query.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0))*self.scale  # scaled dot product

        if mask is not None:
            assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
            attn = attn.masked_fill(mask, -float("inf")).type_as(attn)
            # attn = attn.masked_fill(mask, -1e6).type_as(attn)

        if not self.RFL:
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.relu(attn)

        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        if self.RFL:
            attn_vec=self.norm(attn_vec)
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output

class GatedTransformerLayer(torch.nn.Module):
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
            activation: nn.Module,
            gru_gating: bool = True,
            gru_bias: float = 2.
    ) -> None:
        super(GatedTransformerLayer, self).__init__()
        self.dropout = dropout
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(input_dim, gru_bias)
            self.gate2 = GRUGatingUnit(input_dim, gru_bias)
        self.attention = Attention(
            input_dim,
            head_dim,
            head_num,
            dropout,
        )
        layers = []
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        for i in range(mlp_num):
            layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                activation)
            )
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.activation = activation
    def forward(
            self,
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x1 = self.layernorm1(inputs)
        a1 = self.dropout(self.attention(x1, mask=mask,key_padding_mask=key_padding_mask))
        a1 = self.activation(a1)  # RELU after attention
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
        x2 = self.layernorm2(o1)
        m2 = self.dropout(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2

class GTr(nn.Module):
    def __init__(
            self,
            input_dim: int,
            head_dim: int = 128,
            embedding_dim: int = 256,
            head_num: int = 2,
            mlp_num: int = 2,
            layer_num: int = 3,
            dropout_ratio: float = 0.,
            activation: nn.Module = nn.ReLU(),
            gru_gating: bool = True,
            gru_bias: float = 2.,
            use_embedding_layer: bool = True,
    ) -> None:
        super(GTr, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                activation
            )
        self.activation = activation
        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                GatedTransformerLayer(
                    dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout, self.activation, gru_gating,
                    gru_bias
                )
            )
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, batch_first: bool = False,attn_mask: torch.Tensor=None,key_padding_mask: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        if batch_first:
            x = torch.transpose(x, 1, 0)  # bs x cur_seq x input_dim -> cur_seq x bs x input_dim
        cur_seq, bs = x.shape[:2]

        if self.use_embedding_layer:
            x = self.dropout(self.embedding(x))
        out = x
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out, mask=attn_mask,key_padding_mask=key_padding_mask)

        out = self.dropout(out)
        if batch_first:
            out = torch.transpose(out, 1, 0)  # cur_seq x bs x embedding_dim -> bs x cur_seq x embedding_dim

        output = {"logit": out}
        return output


if __name__ == "__main__":
    model = GTr(input_dim=7, head_num=1, layer_num=1, head_dim=64,embedding_dim=64)

    t = torch.rand(3, 1, 7)

    key_padding_mask= None#torch.randint(0,2,(1,3))

    attn_mask = (
        torch.triu(
            torch.ones((3, 3)),
            diagonal=1,  # fixed in train, eval, collect
        ).bool().unsqueeze(-1)#.to(x.device)
    )  # cur_seq x full_seq x 1

    out=model(t,attn_mask=attn_mask,key_padding_mask=key_padding_mask)

    print(out["logit"].size())


