import torch
import pickle
import logging
import numpy as np
import gym

from typing import Dict, List

from config.config import q_transformer_config
from transformer.q_transformer import QTransformer, ModelArgs
from env.mocks import MockEnvironment

def run(config: Dict[str, str]):

    log = logging.getLogger("Q-Transformer")

    env_name = config['env']
    dataset = config['dataset']
    mode = config['mode']
    pct_traj = config['pct_traj']

    args = ModelArgs()

    env = MockEnvironment(
    state_shape = (3, 6, 224, 224),
    text_embed_shape = (768,))


    model = QTransformer(
        num_actions = 8,
        action_bins = 256,
        depth = 1,
        heads = 8,
        dim_head = 64,
        cond_drop_prob = 0.2,
        dueling = True)

    

if __name__ == '__main__':
    config = q_transformer_config
    run(config = config)