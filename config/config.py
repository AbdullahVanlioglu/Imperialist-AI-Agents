import argparse

decision_config = dict(
    env = 'hopper',
    dataset = 'medium', # medium, medium-replay, medium-expert, expert
    mode = 'normal', # normal for standard setting, delayed for sparse
    K = 20,
    pct_traj = 1.,
    batch_size = 64,
    model_type = 'dt', # dt for decision transformer, bc for behavior cloning
    embed_dim = 128,
    n_layer = 3,
    n_head = 1,
    activation_function = 'relu',
    dropout = 0.1,
    learning_rate = 1e-4,
    weight_decay = 1e-4,
    warmup_steps = 10000,
    num_eval_episodes = 100,
    max_iters = 10,
    num_steps_per_iter = 10000,
    log_to_wandb = '-w',
    device = 'cuda',

)

