import autonomous_optimizer
from stable_baselines3.common import vec_env, monitor
import benchmark
import torch 
import supersuit as ss 
import copy

import numpy as np
import torch
from pettingzoo.utils import wrappers
import tqdm

import matplotlib.pyplot as plt

# from ray import tune

import stable_baselines3
from stable_baselines3.common import vec_env, monitor
import wandb
from wandb.integration.sb3 import WandbCallback

import autonomous_optimizer
import benchmark
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.85)
parser.add_argument('--activation', type=str, default='tanh')
parser.add_argument('--featdim', type=int, default=64)

activation_dict = {
    'relu': torch.nn.ReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid(),
}

args = parser.parse_args()
config = {
    "policy_type": "MlpPolicy",
    "episodes": 200,
    "env_name": "convex_quadratic",
    "num_vars": 20,
    "lr": args.lr,
    "gamma": args.gamma,
    "activation": args.activation,
    'featdim': args.featdim,
}
experiment_name = f"PPO_{int(time.time())}"
# run = wandb.init(
#     project="l2o",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     monitor_gym=False,  # auto-upload the videos of agents playing the game
#     save_code=True,  # optional
# )


quadratic_dataset = [benchmark.convex_quadratic(num_vars=config['num_vars'])  for _ in range(90)]
env = autonomous_optimizer.MARLEnv(quadratic_dataset, num_steps=40, history_len=25)



# env = wrappers.AssertOutOfBoundsWrapper(env)
# env = wrappers.OrderEnforcingWrapper(env)
env = ss.pettingzoo_env_to_vec_env_v0(ss.multiagent_wrappers.pad_action_space_v0(ss.multiagent_wrappers.pad_observations_v0(env)))
# env = ss.concat_vec_envs_v0(env, 8, num_cpus=8, base_class='stable_baselines3')
env = ss.concat_vec_envs_v0(env, 8, num_cpus=8, base_class='stable_baselines3')

policy_kwargs = dict(activation_fn = torch.nn.Tanh,
    net_arch = [dict(pi = [config['featdim'], config['featdim']], vf = [config['featdim'], config['featdim']])])
quadratic_policy = stable_baselines3.PPO(config['policy_type'], env, learning_rate=config['lr'], gamma=config['gamma'],
                        n_steps=2, verbose=0, policy_kwargs = policy_kwargs, tensorboard_log=f"runs/{experiment_name}")

quadratic_policy.learn(total_timesteps=config['episodes'] * 40 * len(quadratic_dataset))
        #  callback=WandbCallback(
        #     model_save_freq=1000,
        #     model_save_path=f"models/{experiment_name}"))

# import pdb; pdb.set_trace()
# obs = quadratic_env.reset()

# quadratic_env.step({0: torch.zeros()})