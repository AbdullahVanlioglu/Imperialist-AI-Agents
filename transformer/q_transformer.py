import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Union, Dict, Any, Tuple, Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 8
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the K and V
    vocab_size: int = -1 # This will be set during to load the tokenizer
    multiple_of: int = 256 # FeedForward multiple of parameter
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    device: str = None

class AttentionBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        
        
    


class QTransformer(nn.Module):
    def __init__(self,
                 state_dim: int,
                 act_dim: int,
                 max_ep_len: int,
                 args: ModelArgs
                 ):
    
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_ep_len = max_ep_len
        self.args = args

        self.transformer = nn.Module()
        for layer_id in range(self.args.n_layers):
            self.layers.append(AttentionBlock(self.args))

    def forward(self, state: torch.Tensor, action: torch.Tensor):

        batch_size, seq_lenght = state.shape[0], state.shape[1]

        


    

