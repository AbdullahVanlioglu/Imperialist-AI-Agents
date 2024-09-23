import torch
import math
import torch.nn as nn
from typing import Any, Dict, Optional

class Attention(nn.Module):
    def __init__(self,
                 query_dim:int,
                 cross_attention_dim:int,
                 heads:int=8,
                 ):
        super().__init__()

        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim
        self.heads = heads


class TransformerBlock(nn.Module):
    def __init__(self,
                 inner_dim:int,
                 num_attention_heads:int,
                 attention_head_dim:int,
                 activation_fn:str,
                 dropout:float=0.0,
                 norm_eps:float=1e-6,
                 ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.activation_fn = activation_fn

        self.norm1 = nn.LayerNorm(inner_dim, eps=norm_eps)

        self.attn1 = Attention()

    def forward(self, x):


class PatchEmbed(nn.Module):
    def __init__(self,
                 height:int,
                 width:int,
                 patch_size:int,
                 layer_norm:bool=False,
                 flatten_output:bool=True,
                 norm_type:str="layer_norm",
                 ):
        super().__init__()

        self.flatten = flatten_output
        self.layer_norm = layer_norm

        num_patches = (height // patch_size) * (width // patch_size)

    def forward(self, latent):
        raise NotImplementedError
        

        


class DiffusionTransformer2D(nn.Module):
    def __init__(self,
                 height:int,
                 width:int,
                 patch_size:int,
                 config:dict,
                 layer_norm:bool=False,
                 flatten_output:bool=True,
                 ):
        super().__init__()

        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        
        self.patch_embed = PatchEmbed(
            height=height: int,
            width=width: int,
            patch_size=patch_size: int,
            layer_norm=layer_norm: bool,
            flatten_output=flatten_output: bool,
        )

    def forward(self, 
                latent:torch.Tensor,
                timestep: Optional[torch.LongTensor]=None,
                class_labels: Optional[torch.Tensor]=None,
                ):

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                inner_dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_function,
            ) for _ in range(self.config.num_layers)
        ])

