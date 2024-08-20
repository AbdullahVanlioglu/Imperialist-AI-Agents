import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, 
                 in_proj_bias: bool = True,
                 out_proj_bias: bool = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self, x: torch.Tensor, casual_mask: bool = False):
        # x: (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> 3 * (batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Head, Seq_Len, Dim/Head)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (Batch_Size, Head, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, Head, Seq_Len, Seq_Len) @ (Batch_Size, Head, Seq_Len, Dim/Head) -> (Batch_Size, Head, Seq_Len, Dim/Head)
        output = weight @ v

        # (Batch_Size, Head, Seq_Len, Dim/Head) -> (Batch_Size, Seq_Len, Head, Dim/Head)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output
    

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, 
                 d_cross: int ,
                 in_proj_bias: bool = True,
                 out_proj_bias: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self, x, y):
        # x: 