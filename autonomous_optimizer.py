"""Main module defining the autonomous RL optimizer."""
import copy

import gym
import numpy as np
import torch
from gym import spaces
from torch import optim

def make_observation(obj_value, obj_values, gradients, num_params_lst, history_len):
    # Features is a matrix where the ith row is a concatenation of the difference
    # in the current objective value and that of the ith previous iterate as well
    # as the ith previous gradient.
    obss = {}
    i = 0
    for agent in num_params_lst.keys():
        num_params = num_params_lst[agent]
        observation = np.zeros((history_len, 1+num_params), dtype="float32")
        try:
            for j in range(len(obj_values)):
                observation[j, 0] = obj_value - obj_values[j].cpu().numpy()
                observation[j, 1:] = gradients[j][i].cpu().numpy()
        except:
            import pdb; pdb.set_trace()
        observation /= 50        
        observation.clip(-1, 1)
        obss[agent] = observation
        i += 1
    # Normalize and clip observation space
    return obss

class AutonomousOptimizer(optim.Optimizer):
    def __init__(self, params, policy, history_len=25):
        """
        Parameters:
            policy: Policy that takes in history of objective values and gradients
                as a feature vector - shape (history_len, num_parameters + 1),
                and outputs a vector to update parameters by of shape (num_parameters,).
            history_len: Number of previous iterations to keep objective value and
                gradient information for.

        """
        super().__init__(params, {})

        self.policy = policy
        self.history_len = history_len
        self.num_params = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )

        self.obj_values = []
        self.gradients = []

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            obj_value = closure()

        # Calculate the current gradient and flatten it
        current_grad = torch.cat(
            [p.grad.flatten() for group in self.param_groups for p in group["params"]]
        ).flatten()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)

        # Run policy
        observation = make_observation(
            obj_value.item(),
            self.obj_values,
            self.gradients,
            self.num_params,
            self.history_len,
        )
        action, _states = self.policy.predict(observation, deterministic=True)

        # Update the parameters according to the policy
        param_counter = 0
        action = torch.from_numpy(action)
        for group in self.param_groups:
            for p in group["params"]:
                delta_p = action[param_counter : param_counter + p.numel()]
                p.add_(delta_p.reshape(p.shape))
                param_counter += p.numel()

        return obj_value


class Environment(gym.Env):
    """Optimization environment based on TF-Agents."""

    def __init__(
        self,
        dataset,
        num_steps,
        history_len,
    ):
        super().__init__()

        self.dataset = dataset
        self.num_steps = num_steps
        self.history_len = history_len

        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_params,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.history_len*2, self.num_params),
            dtype=np.float32,
        )
        # self.observation_space = spaces.Box(
        #     low=-1,
        #     high=1,
        #     shape=(self.history_len, 1 + self.num_params),
        #     dtype=np.float32,
        # )

    def _setup_episode(self):
        res = np.random.choice(self.dataset)
        self.model = copy.deepcopy(res["model0"])
        self.obj_function = res["obj_function"]

        self.obj_values = []
        self.gradients = []
        self.current_step = 0

    def reset(self):
        self._setup_episode()
        return make_observation(
            None, self.obj_values, self.gradients, self.num_params, self.history_len
        )

    @torch.no_grad()
    def step(self, action):
        # Update the parameters according to the action
        action = torch.from_numpy(action)
        param_counter = 0
        for p in self.model.parameters():
            delta_p = action[param_counter : param_counter + p.numel()]
            p.add_(delta_p.reshape(p.shape))
            param_counter += p.numel()

        # Calculate the new objective value
        with torch.enable_grad():
            self.model.zero_grad()
            obj_value = self.obj_function(self.model)
            obj_value.backward()

        # Calculate the current gradient and flatten it
        current_grad = torch.cat(
            [p.grad.flatten() for p in self.model.parameters()]
        ).flatten()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)

        # Return observation, reward, done, and empty info
        observation = make_observation(
            obj_value.item(),
            self.obj_values,
            self.gradients,
            self.num_params,
            self.history_len,
        )
        reward = -obj_value.item()
        done = self.current_step >= self.num_steps
        info = {}

        self.current_step += 1
        return observation, reward, done, info

import functools
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import from_parallel

class MARLEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "marl_v1"}
    def __init__(self, dataset, num_steps, history_len):
        super().__init__()

        self.dataset = dataset
        self.num_steps = num_steps
        self.history_len = history_len

        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        # Define action and observation space
        
        self.num_params_lst, self.possible_agents = {}, []
        
        i = 0
        for p in self.model.parameters():
            self.possible_agents.append('agent_'+str(i))
            self.num_params_lst[self.possible_agents[i]] = p.numel()
            i += 1
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.action_spaces, self.observation_spaces = {}, {}
        for agent in self.possible_agents:
            self.observation_spaces[agent] = spaces.Box(low=-1, high=1, shape=(self.history_len, self.num_params_lst[agent]+1))
            self.action_spaces[agent] = spaces.Box(low=-1, high=1, shape=(self.num_params_lst[agent],), dtype=np.float32) 

    
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return spaces.Box(low=-1, high=1, shape=(self.history_len, self.num_params_lst[agent]+1))
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(self.num_params_lst[agent],), dtype=np.float32)
    
    def render(self, mode="human"):
        pass

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def _setup_episode(self):
        res = np.random.choice(self.dataset)
        self.model = copy.deepcopy(res["model0"])
        self.obj_function = res["obj_function"]

        self.obj_values = []
        self.gradients = []
        self.current_step = 0

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        '''
        self.agents = self.possible_agents[:]
        self._setup_episode()
        obss = make_observation(
            None, self.obj_values, self.gradients, self.num_params_lst, self.history_len
        )
        self.current_step = 0
        return obss

    @torch.no_grad()
    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        param_counter = 0
        for p in self.model.parameters():
            delta_p = torch.Tensor(actions[self.possible_agents[param_counter]]).cuda()
            p.add_(delta_p.reshape(p.shape))
            param_counter += 1
        
        # Calculate the new objective value
        with torch.enable_grad():
            self.model.zero_grad()
            obj_value = self.obj_function(self.model)
            obj_value.backward()

        # Calculate the current gradient and flatten it
        current_grad =[p.grad.flatten() for p in self.model.parameters()]

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)
        # Return observation, reward, done, and empty info
        observations = make_observation(
            obj_value.item(),
            self.obj_values,
            self.gradients,
            self.num_params_lst,
            self.history_len,
        )
        rewards = {}
        dones = {}
        env_done = self.current_step >= self.num_steps
        for agent in self.agents:
            rewards[agent] = -obj_value.item()
            dones[agent] = env_done

        
        self.current_step += 1
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos
        

def env(**kwargs):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = MARLEnv(**kwargs)

    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

