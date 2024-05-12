import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Dict, Any, Tuple



class QTransformer(nn.Module):
    def __init__(self,
                 num_actions: int = 8,
                 action_bins:int = 256,
                 depth:int = 6,
                 heads:int = 8,
                 dim_head: int = 64,
                 token_learner_ff_mult:int = 2,
                 token_learner_num_layers:int = 2,
                 token_learner_num_output_tokens:int = 8,
                 cond_drop_prob:float = 0.2,
                 use_attn_conditioner:bool = False,
                 conditioner_kwargs:dict = dict(),
                 dueling: bool = False,
                 flash_attn: bool = True,
                 condition_on_text: bool = True,
                 q_head_attn_kwargs: dict = dict(
                     attn_heads = 8,
                     attn_dim_head = 64,
                     attn_depth = 2
                 ),
                 weight_tie_action_bid_embed = True
                 ):
        super().__init__()

