import torch
import pickle
import logging
import numpy as np
import gym

from typing import Dict, List

from config.config import q_transformer_config
from train.trainer import QTrainer
from transformer.q_transformer import QTransformer, ModelArgs


def run(config: Dict[str, str]):

    args = ModelArgs()
    num_actions = args["num_actions"]
    action_bins = args["action_bins"]
    depth = args["depth"]
    heads = args["heads"]
    dim_head = args["dim_head"]
    cond_drop_prob = args["cond_drop_prob"]
    

    # Inputs
    video = torch.randn(2, 3, 6, 224, 224)
    instructions = [
        'bring me that apple sitting on the table',
        'please pass the butter']



    model = QTransformer(
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