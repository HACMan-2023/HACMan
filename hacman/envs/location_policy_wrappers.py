import os, sys

import numpy as np
import gym

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs
from hacman.utils.plotly_utils import plot_pcd, plot_action, plot_pcd_with_score
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from hacman.algos.location_policy import RandomLocation


class SubprocVecEnvwithLocationPolicy(SubprocVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], 
                 start_method: Optional[str] = None, 
                 location_model: Callable[[Dict], Dict] = None):
        
        self.location_model = location_model
        super().__init__(env_fns, start_method)
    
    def process_obs(self, obs):
        count = 0
        for sub_obs in obs:
            location_info = self.location_model.get_action(sub_obs)
            sub_obs.update(location_info)
            assert self.env_method('set_prev_obs', sub_obs, indices=count)
            count += 1
        return
    
    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        self.process_obs(obs) # New
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos
    
    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        self.process_obs(obs) # New
        return _flatten_obs(obs, self.observation_space)


class DummyVecEnvwithLocationPolicy(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]],
                 location_model: Callable[[Dict], Dict] = None):
        
        self.location_model = location_model
        super().__init__(env_fns)
    
    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        location_info = self.location_model.get_action(obs)
        obs.update(location_info)
        assert self.env_method('set_prev_obs', obs, indices=env_idx)
        return super()._save_obs(env_idx, obs)
