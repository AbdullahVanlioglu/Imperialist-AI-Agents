import gymnasium as gym
import numpy as np
import torch
import random
import pickle
import logging
import wandb

from dataclasses import dataclass
from config.config import decision_config
from typing import Optional, List

from transformer.decision_transformer import DecisionTransformer
from transformer.behavior_clonning import MLBehaviorClonning
from train.trainer import SequenceTrainer, ActTrainer


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum



def experiment(exp, config):

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
    traj_lengths, returns = np.array(traj_lens), np.array(returns)

    # Used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lengths)

    logging.warning(f'Starting new experiment: {env_name} {dataset}')
    logging.warning(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    logging.warning(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logging.warning(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')

    K = config['K']
    batch_size = config['batch_size']
    num_eval_episodes = config['num_eval_episodes']
    pct_traj = config.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lengths[sorted_inds[-1]]
    ind = len(trajectories) - 2

    while ind >= 0 and timesteps + traj_lengths[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lengths[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lengths[sorted_inds] / sum(traj_lengths[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample)  # reweights so we sample according to timesteps

        batch_state, batch_action, batch_reward, batch_done, batch_rtg, batch_timestep, batch_mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            batch_state.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            batch_action.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            batch_reward.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                batch_done.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                batch_done.append(traj['dones'][si:si + max_len].reshape(1, -1))
            batch_timestep.append(np.arange(si, si + batch_state[-1].shape[1]).reshape(1, -1))
            batch_timestep[-1][batch_timestep[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            batch_rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:batch_state[-1].shape[1] + 1].reshape(1, -1, 1))
            if batch_rtg[-1].shape[1] <= batch_state[-1].shape[1]:
                batch_rtg[-1] = np.concatenate([batch_rtg[-1], np.zeros((1, 1, 1))], axis=1)
    
            # padding and state + reward normalization
            tlen = batch_state[-1].shape[1]
            batch_state[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), batch_state[-1]], axis=1)
            batch_state[-1] = (batch_state[-1] - state_mean) / state_std
            batch_action[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., batch_action[-1]], axis=1)
            batch_reward[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), batch_reward[-1]], axis=1)
            batch_done[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, batch_done[-1]], axis=1)
            batch_rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), batch_rtg[-1]], axis=1) / scale
            batch_timestep[-1] = np.concatenate([np.zeros((1, max_len - tlen)), batch_timestep[-1]], axis=1)
            batch_mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        batch_state = torch.from_numpy(np.concatenate(batch_state, axis=0)).to(dtype=torch.float32, device=device)
        batch_action = torch.from_numpy(np.concatenate(batch_action, axis=0)).to(dtype=torch.float32, device=device)
        batch_reward = torch.from_numpy(np.concatenate(batch_reward, axis=0)).to(dtype=torch.float32, device=device)
        batch_done = torch.from_numpy(np.concatenate(batch_done, axis=0)).to(dtype=torch.long, device=device)
        batch_rtg = torch.from_numpy(np.concatenate(batch_rtg, axis=0)).to(dtype=torch.float32, device=device)
        batch_timestep = torch.from_numpy(np.concatenate(batch_timestep, axis=0)).to(dtype=torch.long, device=device)
        batch_mask = torch.from_numpy(np.concatenate(batch_mask, axis=0)).to(device=device)

        return batch_state, batch_action, batch_reward, batch_done, batch_rtg, batch_timestep, batch_mask
    
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=config['embed_dim'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=4*config['embed_dim'],
            activation_function=config['activation_function'],
            n_positions=1024,
            resid_pdrop=config['dropout'],
            attn_pdrop=config['dropout'],
        )
    elif model_type == 'bc':
        model = MLBehaviorClonning(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=config['embed_dim'],
            n_layer=config['n_layer'],
        )
    else:
        raise NotImplementedError
    
    model = model.to(device=device)

    warmup_steps = config['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1))
    
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets])
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets])
        
    if log_to_wandb:
        wandb.init(
            name=exp,
            group=group_name,
            project='decision-transformer',
            config=config
        )

    for iter in range(config['max_iters']):
        outputs = trainer.train_iteration(num_steps=config['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
    


if __name__ == '__main__':
    config = decision_config
    experiment('gym-experiment', config = config)