import copy
import autonomous_optimizer
from stable_baselines3.common import vec_env, monitor
import benchmark
import torch 
import supersuit as ss 
import copy

import numpy as np
import torch
from pettingzoo.utils import wrappers
import numpy as np
import torch
import tqdm

import matplotlib.pyplot as plt
import multiprocessing as mp
# from ray import tune

import stable_baselines3
from stable_baselines3.common import vec_env, monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common import vec_env, monitor

import autonomous_optimizer
import benchmark
import argparse
import time
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

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
    "num_vars": 200,
    "num_steps": 40,
    "lr": args.lr,
    "gamma": args.gamma,
    "activation": args.activation,
    'featdim': args.featdim,
}

class TensorboardCallback(WandbCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0, model_save_path: str = None, model_save_freq: int = 0, gradient_save_freq: int = 0,):
        super(TensorboardCallback, self).__init__(verbose, model_save_path, model_save_freq, gradient_save_freq)
        self.num_episode = 0 
        
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.rewards = np.zeros_like(self.training_env.venv.venv.shared_rews.shared_arr)
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        wandb.log({'episode/avg_rewards': np.mean(self.rewards)}, step=self.num_episode)
        wandb.log({'episode/std_rewards': np.std(self.rewards)}, step=self.num_episode)
        self.rewards = np.zeros_like(self.training_env.venv.venv.shared_rews.shared_arr)
        self.num_episode += 1

    def _on_step(self):
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        self.rewards += np.array(self.training_env.venv.venv.shared_rews.shared_arr)
        self.dones = np.array(self.training_env.venv.venv.shared_dones.shared_arr)
        # import pdb; pdb.set_trace()
  
        
        return True

if __name__ == '__main__':
    # ctx_in_main = mp.get_context('forkserver')
    # ctx_in_main.set_forkserver_preload(['inherited'])
    mp.set_start_method('spawn')
    experiment_name = f"PPO_{int(time.time())}"
    run = wandb.init(
        project="l2o-marl",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gym 
    from torch import nn

    policy_kwargs = dict(activation_fn = torch.nn.Tanh,
        net_arch = [dict(pi = [config['featdim'], config['featdim']], vf = [config['featdim'], config['featdim']])])

    quadratic_dataset = [benchmark.convex_quadratic_joint(num_vars=config['num_vars']) for _ in range(360)]

    env = autonomous_optimizer.MARLEnv(quadratic_dataset, num_steps=config['num_steps'], history_len=25)
    env = ss.pettingzoo_env_to_vec_env_v0(ss.multiagent_wrappers.pad_action_space_v0(ss.multiagent_wrappers.pad_observations_v0(env)))
    env = ss.concat_vec_envs_v0(env, 10, num_cpus=20, base_class='stable_baselines3')

    quadratic_env = VecMonitor(env)

    quadratic_policy = stable_baselines3.PPO(config['policy_type'], quadratic_env, learning_rate=config['lr'], gamma=config['gamma'],
                            n_steps=2, verbose=0, policy_kwargs = policy_kwargs, tensorboard_log=f"runs/{experiment_name}")

    quadratic_policy.learn(total_timesteps=config['episodes'] * config['num_steps'] * len(quadratic_dataset),
            callback=TensorboardCallback(
                model_save_freq=1000,
                model_save_path=f"models/{experiment_name}"))

    quadratic_tune = {
        "sgd": {"hyperparams": {"lr": 5e-2}},
        "momentum": {"hyperparams": {"lr": 1e-2, "momentum": 0.7}},
        "adam": {"hyperparams": {"lr": 1e-1}},
        "lbfgs": {"hyperparams": {"lr": 1, "max_iter": 1}}
    }

    problem = benchmark.convex_quadratic_joint(num_vars=config['num_vars'])

    model0 = problem["model0"]
    obj_function = problem["obj_function"]
    optimal_x = problem["optimal_x"]
    optimal_value = problem["optimal_val"]
    A = problem["A"]
    b = problem["b"]

    print(f'Objective function minimum: {optimal_value}')

    iterations = 40

    results = benchmark.run_all_optimizers(problem, iterations, quadratic_tune, quadratic_policy)
    sgd_vals, sgd_traj = results["sgd"]
    momentum_vals, momentum_traj = results["momentum"]
    adam_vals, adam_traj = results["adam"]
    lbfgs_vals, lbfgs_traj = results["lbfgs"]
    ao_vals, ao_traj = results["ao"]
    wandb.log({"gap": abs(ao_vals[-1]-optimal_value)})

    xs = [i for i in range(len(sgd_vals))]
    ys = []

    for my_val in [sgd_vals, momentum_vals, adam_vals, lbfgs_vals, ao_vals]:
        ys.append((my_val - optimal_value).tolist())
    keys=["SGD", "Momentum", "Adam","LBFGS", "Autonomous Optimizer"]

    wandb.log({"object_value" : wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title="Convex Quadratic objective value",
            xname="Iteration")})
