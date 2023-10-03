import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.preprocessing import preprocess_obs as pobs
from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy, Actor


class MixTD3Policy(TD3Policy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        actor_features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        critic_features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        actor_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        critic_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        self.actor_features_extractor_class = actor_features_extractor_class
        self.critic_features_extractor_class = critic_features_extractor_class
        if actor_features_extractor_kwargs is None:
            actor_features_extractor_kwargs = {}
        if critic_features_extractor_kwargs is None:
            critic_features_extractor_kwargs = {}
        self.actor_features_extractor_kwargs = actor_features_extractor_kwargs
        self.critic_features_extractor_kwargs = critic_features_extractor_kwargs
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=None,
            features_extractor_kwargs=None,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,            
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self.actor_kwargs.copy()
        features_extractor = self.actor_features_extractor_class(self.observation_space, **self.actor_features_extractor_kwargs)
        actor_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))    
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:        
        critic_kwargs = self.critic_kwargs.copy()
        features_extractor = self.critic_features_extractor_class(self.observation_space, **self.critic_features_extractor_kwargs)
        critic_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))    
        return ContinuousCritic(**critic_kwargs).to(self.device)
