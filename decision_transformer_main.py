import gym
import numpy as np
import torch
import random
import argparse
import pickle
import logging

from dataclasses import dataclass
from config import decision_config
from typing import Optional, List


@dataclass
class ModelArgs:
    hopper_max_ep_len: int = 1000,
    hopper_env_targets: List = [3600, 1800],
    hopper_scale: float = 1000.

    halfcheetah_max_ep_len:int = 1000,
    halfcheetah_env_targets: List = [12000, 6000],
    halfcheetah_scale:float = 1000.



def experiment(
        exp,
        config,
):
    device = config.get('device', 'cuda')
    log_to_wandb = config.get('log_to_wandb', False)

    env_name, dataset = config['env'], config['dataset']
    model_type = config['model_type']
    group_name = f'{exp}-{env_name}-{dataset}'
    exp = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

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
    
    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'datasets/gym/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = config.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # Used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    logging.info('-' * 50)
    logging.info(f'Starting new experiment: {env_name} {dataset}')
    logging.info(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    logging.info(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logging.info(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    logging.info('-' * 50)

    K = config['K']
    batch_size = config['batch_size']
    num_eval_episodes = config['num_eval_episodes']
    pct_traj = config.get('pct_traj', 1.)
    



    
    











if __name__ == '__main__':
    config = decision_config

    experiment('gym-experiment', config = config)