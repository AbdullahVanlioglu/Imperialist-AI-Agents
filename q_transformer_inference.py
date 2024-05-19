import torch
import pickle
import numpy as np

from config.config import q_transformer_config

config = q_transformer_config
env_name = config['env_name']
dataset = config['dataset']
mode = config['mode'] # Delayed Reward

# load dataset
dataset_path = f'datasets/gym/{env_name}-{dataset}-v2.pkl'
with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)


states, traj_lens, returns = [], [], []
for path in trajectories:
    path['rewards'][-1] = path['rewards'].sum()
    path['rewards'][:-1] = 0.
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum())

traj_lengths, returns = np.array(traj_lens), np.array(returns)