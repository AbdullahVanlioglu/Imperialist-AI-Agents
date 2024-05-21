import torch
import pickle
import logging
import numpy as np
import gym

from typing import Dict, List

from config.config import q_transformer_config
from transformer.q_transformer import QTransformer, ModelArgs
from transformer.maxvit import MaxViT


def run(config: Dict[str, str]):

    args = ModelArgs()
    num_actions = args["num_actions"]
    action_bins = args["action_bins"]
    depth = args["depth"]
    heads = args["heads"]
    dim_head = args["dim_head"]
    cond_drop_prob = args["cond_drop_prob"]
    
    # Inputs
    img = torch.randn(1, 3, 256, 256)
    text = torch.randint(0, 20000, (1, 1024))

    vision_transformer = MaxViT.max_vit_base_224(in_channels=4)
    output = vision_transformer(img)

    breakpoint()

    model = QTransformer(
        vit = vision_transformer,
        num_actions = num_actions,
        action_bins = action_bins,
        depth = depth,
        heads = heads,
        dim_head = dim_head,
        cond_drop_prob = cond_drop_prob,
        dueling = True)


if __name__ == '__main__':
    config = q_transformer_config
    run(config = config)