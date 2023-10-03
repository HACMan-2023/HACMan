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


class HACManTD3(TD3):
    def __init__(self, *args, temperature=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        
    def get_next_q_values(self, replay_data):
        obs_tensor = replay_data.next_observations
        batch_size = obs_tensor['object_pcd_points'].shape[0]
        object_pcd_size = obs_tensor['object_pcd_points'].shape[1]
        preprocessed_obs = pobs(obs_tensor, self.actor.observation_space, normalize_images=self.actor.normalize_images)
        
        # Get per point action only for object points
        features = self.actor_target.features_extractor(preprocessed_obs, per_point_output=True)
        features = features.reshape(batch_size, -1, features.shape[-1])[:, :object_pcd_size, :]
        actions = self.actor_target.mu(features)

        # Add noise        
        noise = actions.clone().data.normal_(0, self.target_policy_noise)
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        actions = (actions + noise).clamp(-1, 1)
        
        # Get per point feature
        features = self.critic_target.features_extractor(preprocessed_obs, per_point_output=True)
        features = features.reshape(batch_size, -1, features.shape[-1])[:, :object_pcd_size, :]
        # Concatenate with per point critic feature and get per point value
        out1 = self.critic_target.q_networks[0](th.cat([features, actions], dim=2)).reshape(batch_size, -1, 1)
        out2 = self.critic_target.q_networks[1](th.cat([features, actions], dim=2)).reshape(batch_size, -1, 1)
        next_q_values = th.cat([out1, out2], dim=2)
        
        if self.mean_q:
            next_q_values = th.mean(next_q_values, dim=2, keepdim=True)
        else:
            next_q_values, _ = th.min(next_q_values, dim=2, keepdim=True)
        
        # object_pcd_points dim: batch x pcd_size x 3
        action_score = th.nn.Softmax(dim=1)(next_q_values/self.temperature)
        next_q_values = th.sum(next_q_values*action_score, dim=1)
            
        return next_q_values
    
    def get_actor_loss(self, replay_data):
        obs_tensor = replay_data.observations
        batch_size = obs_tensor['object_pcd_points'].shape[0]
        object_pcd_size = obs_tensor['object_pcd_points'].shape[1]
        preprocessed_obs = pobs(obs_tensor, self.actor.observation_space, normalize_images=self.actor.normalize_images)
        
        # Get per point action only for object points
        features = self.actor.features_extractor(preprocessed_obs, per_point_output=True)
        features = features.reshape(batch_size, -1, features.shape[-1])[:, :object_pcd_size, :]
        actions = self.actor.mu(features)
        
        # Get per point feature
        features = self.critic.features_extractor(preprocessed_obs, per_point_output=True)
        features = features.reshape(batch_size, -1, features.shape[-1])[:, :object_pcd_size, :]
        # Concatenate with per point critic feature and get per point value
        qvalue = self.critic.q_networks[0](th.cat([features, actions], dim=2)).reshape(batch_size, -1, 1)
        action_score = th.nn.Softmax(dim=1)(qvalue/self.temperature)
        
        actor_loss = -th.sum(qvalue*action_score, dim=1).mean()
        return actor_loss

