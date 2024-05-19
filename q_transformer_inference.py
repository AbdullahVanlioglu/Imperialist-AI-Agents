import torch
import pickle
import logging
import numpy as np
import gym

from typing import Dict, List

from config.config import q_transformer_config
from transformer.q_transformer import QTransformer, ModelArgs

def run(config: Dict[str, str]):

    env_name = config['env']
    dataset = config['dataset']
    mode = config['mode']

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from environment.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError


    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]


    args = ModelArgs()


    # load dataset
    dataset_path = f'datasets/gym/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)


    states, traj_lengths, returns = [], [], []
    for path in trajectories:
        path['rewards'][-1] = path['rewards'].sum()  # Delayed Reward
        path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lengths.append(len(path['observations']))
        returns.append(path['rewards'].sum())

    traj_lengths, returns = np.array(traj_lengths), np.array(returns)

    # Used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lengths)

    logging.info('-' * 50)
    logging.info(f'Starting new experiment: {env_name} {dataset}')
    logging.info(f'{len(traj_lengths)} trajectories, {num_timesteps} timesteps found')
    logging.info(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logging.info(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    logging.info('-' * 50)

    model = QTransformer(
        state_dim = state_dim, 
        action_dim = act_dim,
        max_ep_len = max_ep_len,
        args = args
        )

    

if __name__ == '__main__':
    config = q_transformer_config
    run(config = config)