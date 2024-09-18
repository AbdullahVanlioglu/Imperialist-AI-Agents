import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Union, Dict, Any, Tuple, Optional
from transformer.maxvit_transformer import MaxViT

@dataclass
class ModelArgs:
    num_actions: int = 8
    action_bins: int = 256
    depth: int = 1
    heads:int = 5
    dim_head:int = 64
    cond_drop_prob:float = 0.2

    embed_dim: int = 512
    n_layers: int = 8
    n_heads: int = 32 # Number of heads for the queries
    max_seq_lenght: int = 120 # Number of sequence lenght
    batch_size: int = 256
    window_size: int = 7 # Number of window 
    n_kv_heads: Optional[int] = None # Number of heads for the K and V
    device: str = None

class AttentionBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args


    
class QTransformer(nn.Module):
    def __init__(self,
                 vit: MaxViT,
                 args: ModelArgs,
                 state_dim: int,
                 act_dim: int,
                 max_ep_len: int
                 ):
    
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_ep_len = max_ep_len
        self.args = args

        self.tok_embeddings = torch.nn.Linear(self.state_dim, self.args.embed_dim)
        self.time_embeddigns = torch.nn.Linear(self.args.max_seq_lenght, self.args.embed_dim)

        self.transformer = nn.Module()
        for layer_id in range(self.args.n_layers):
            self.transformer.append(AttentionBlock(self.args))

    def forward(self, state: torch.Tensor, action: torch.Tensor, timestep: int):
        # (B, Seq_Len, State_Dim)
        batch_size, seq_lenght = state.shape[0], state.shape[1]

        # (B, Seq_Len, State_Dim) -> (B, Seq_Len, State_Dim, Embed_Dim)
        state_emb = self.tok_embeddings(state)
        time_emb = self.time_embeddigns(timestep)

        state_embeddings = state_emb + time_emb

        
