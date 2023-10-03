import os, sys
import gym
import torch
from torch import nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import create_mlp
from hacman.networks.common import init_network
from hacman.utils.transformations import transform_point_cloud, decompose_pose_tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MLP


class PointCloudExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, normalize_pos_param=None, 
                 preprocessing_fn='flow'):
        # Normalize_range is in the form of [offset, scale]
        super().__init__(observation_space, features_dim=128, ) # feature_dim=32 for pt
        config = {"model": "pn",
                  "dropout": 0.,
                  "fps_deterministic": False,
                  "pos_in_feature": False,
                  "normalize_pos": True,
                  "normalize_pos_param": normalize_pos_param}
        if 'batch' in preprocessing_fn:
            # backward compatibility
            preprocessing_fn = preprocessing_fn[:-6]
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing = globals()['preprocessing_'+preprocessing_fn] # Get processing function by name
        input_channels = 1 # Mask channel
        if preprocessing_fn == 'flow' or preprocessing_fn == 'flow_v0':
            input_channels += 3
        elif preprocessing_fn == 'goal_pose' or preprocessing_fn == 'goal_pose_v0':
            input_channels += 12
        elif preprocessing_fn == 'goal_pose_quat':
            input_channels += 7
        self.backbone = init_network(config, input_channels=input_channels, output_channels=[])
        
    def forward(self, observations: TensorDict, per_point_output=False) -> torch.Tensor:
        batch_size = observations['object_pcd_points'].shape[0]

        if 'v0' in self.preprocessing_fn:
            # x2 slower
            data_list = []
            for batch_id in range(batch_size):
                datum = self.preprocessing(observations, batch_id)
                data_list.append(datum)
            data = Batch.from_data_list(data_list)
            out = self.backbone(data.x, data.pos, data.batch)            
        else:
            data_x, data_pos, data_batch = self.preprocessing(observations)
            out = self.backbone(data_x, data_pos, data_batch)
        
        if per_point_output == False:
            out = out.reshape(batch_size, int(out.shape[0]/batch_size), -1)[:, :, :]
            # Only output the feature at poke_idx
            out_list = []
            for batch_id in range(batch_size):
                poke_idx = observations['poke_idx'][batch_id].detach().cpu().numpy()
                out_list.append(out[batch_id, poke_idx, :].reshape(1, -1))
            out = torch.cat(out_list, dim=0)
        
        return out
    
    
class PointCloudGlobalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 preprocessing_fn='flow',
                 include_gripper=False, normalize_pos_param=None):
        # Normalize_range is in the form of [offset, scale]
        feature_dim = 1024
        if include_gripper:
            feature_dim += 7
        super().__init__(observation_space, features_dim=feature_dim)
        config = {"model": "pnc",
                  "dropout": 0.,
                  "fps_deterministic": False,
                  "pos_in_feature": False,
                  "normalize_pos": True,
                  "normalize_pos_param": normalize_pos_param}
        
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing = globals()['preprocessing_'+preprocessing_fn] # Get processing function by name
        input_channels = 1 # Mask channel
        if preprocessing_fn == 'flow' or preprocessing_fn == 'flow_v0':
            input_channels += 3
        elif preprocessing_fn == 'goal_pose' or preprocessing_fn == 'goal_pose_v0':
            input_channels += 12
        elif preprocessing_fn == 'goal_pose_quat':
            input_channels += 7
        self.backbone = init_network(config, input_channels=input_channels, output_channels=[])
        self.include_gripper = include_gripper
        
    def forward(self, observations: TensorDict) -> torch.Tensor:
        batch_size = observations['object_pcd_points'].shape[0]

        if 'v0' in self.preprocessing_fn:
            # x2 slower
            data_list = []
            for batch_id in range(batch_size):
                datum = self.preprocessing(observations, batch_id)
                data_list.append(datum)
            data = Batch.from_data_list(data_list)
            out = self.backbone(data.x, data.pos, data.batch)            
        else:
            data_x, data_pos, data_batch = self.preprocessing(observations)
            out = self.backbone(data_x, data_pos, data_batch)
        
        if self.include_gripper:
            gripper_pose = observations['gripper_pose']
            gripper_pose = decompose_pose_tensor(gripper_pose, cat=True)    # Shape: [N, 3 + 4]
            out = torch.cat([out, gripper_pose], dim=1)
        
        return out


class StatesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, include_gripper=False):
        features_dim = 17
        if include_gripper:
            features_dim += 7
        super().__init__(observation_space, features_dim=features_dim)
        # self.net = create_mlp(32, 128, [64])
        # self.net = MLP([32, 64, 128])
        self.include_gripper = include_gripper

    def forward(self, observations: TensorDict) -> torch.Tensor:
        batch_size = observations['object_pose'].shape[0]
        # for batch_id in range(batch_size):
        #     datum = preprocessing(observations, batch_id)
        #     data_list.append(datum)
        # data = Batch.from_data_list(data_list)
        # out = self.backbone(data.x, data.pos, data.batch)
        object_pose = decompose_pose_tensor(observations['object_pose'], cat=True)
        goal_pose = decompose_pose_tensor(observations['goal_pose'], cat=True)
        object_size = observations['object_size']
        obs_list = [object_pose, goal_pose, object_size]
        
        if self.include_gripper:
            gripper_pose = decompose_pose_tensor(observations['gripper_pose'], cat=True)
            obs_list.append(gripper_pose)
        
        obs = torch.cat(obs_list, dim=1)
        obs = obs.view([batch_size, -1])
        
        return obs


# Input Preprocessing Functions
def preprocessing_flow(obs):
    
    # Flow as point features.
    # Use object pose and goal pose to calculate the flow
    dtype = obs['object_pcd_points'].dtype
    device = obs['object_pcd_points'].device

    # Object points    
    object_pcd_points = obs['object_pcd_points']
    
    transform = obs['goal_pose'] @ torch.linalg.inv(obs['object_pose'])
    # transform = torch.linalg.solve(obs['object_pose'], obs['goal_pose'], left=False) # T*current_pose = goal_pose    
    # faster but requires pytorch 1.13
    
    ones = torch.ones(object_pcd_points.shape[0], object_pcd_points.shape[1], 1, dtype=dtype, device=device)
    padded_points = torch.cat([object_pcd_points, ones], dim=-1)
    flow = torch.matmul(transform, padded_points.transpose(1, 2)).transpose(1, 2)[:,:,:3] - object_pcd_points
    mask = torch.ones((object_pcd_points.shape[0], object_pcd_points.shape[1], 1), dtype=dtype, device=device)
    object_feature = torch.cat([mask, flow], axis=-1)
    
    # Backround points
    background_pcd_points = obs['background_pcd_points']
    background_feature = torch.zeros((background_pcd_points.shape[0], background_pcd_points.shape[1], 4), dtype=dtype, device=device)
                    
    data_pos = torch.cat([object_pcd_points, background_pcd_points], axis=1)
    data_pos = data_pos.reshape(-1, 3)
    data_x = torch.cat([object_feature, background_feature], axis=1)
    data_x = data_x.reshape(-1, data_x.shape[-1])
    data_batch = torch.arange(object_pcd_points.shape[0], device=device).repeat_interleave(object_pcd_points.shape[1] + background_pcd_points.shape[1])
    return data_x, data_pos, data_batch

# Concatenate goal points
def preprocessing_no_flow(obs):
    dtype = obs['object_pcd_points'].dtype
    device = obs['object_pcd_points'].device
    batch_size = obs['object_pcd_points'].shape[0]
    num_obj_points = obs['object_pcd_points'].shape[1]
    num_bg_points = obs['background_pcd_points'].shape[1]

    # Object points    
    object_pcd_points = obs['object_pcd_points']
    
    transform = obs['goal_pose'] @ torch.linalg.inv(obs['object_pose'])
    # transform = torch.linalg.solve(obs['object_pose'], obs['goal_pose'], left=False) # T*current_pose = goal_pose    
    # faster but requires pytorch 1.13
    
    ones = torch.ones(batch_size, num_obj_points, 1, dtype=dtype, device=device)
    padded_points = torch.cat([object_pcd_points, ones], dim=-1)
    transformed_pcd = torch.matmul(transform, padded_points.transpose(1, 2)).transpose(1, 2)[:,:,:3]
    
    # Backround points
    background_pcd_points = obs['background_pcd_points']
                    
    full_points = torch.cat([object_pcd_points, background_pcd_points, transformed_pcd], axis=1)
    
    # Object Mask: Object=1 Backgound=0 Goal=-1
    mask = torch.zeros((batch_size, num_obj_points*2 + num_bg_points, 1), dtype=dtype, device=device)
    mask[:, :num_obj_points] = 1.
    mask[:, num_obj_points+num_bg_points:] = -1.

    # Flatten points across batch
    data_pos = full_points.reshape(-1, 3)
    data_x = mask.reshape(-1, 1)
    data_batch = torch.arange(batch_size, device=device).repeat_interleave(num_obj_points*2 + num_bg_points)
    return data_x, data_pos, data_batch


def preprocessing_goal_pose_quat(obs):
    return preprocessing_goal_pose(obs, quat=True)


def preprocessing_goal_pose(obs, quat=False):
    # Concatenate target transformation in point features
    dtype = obs['object_pcd_points'].dtype
    device = obs['object_pcd_points'].device
    batch_size = obs['object_pcd_points'].shape[0]
    num_obj_points = obs['object_pcd_points'].shape[1]
    num_bg_points = obs['background_pcd_points'].shape[1]

    # Object points    
    object_pcd_points = obs['object_pcd_points']
    
    transform = obs['goal_pose'] @ torch.linalg.inv(obs['object_pose'])
    # transform = torch.linalg.solve(obs['object_pose'], obs['goal_pose'], left=False) # T*current_pose = goal_pose    
    # faster but requires pytorch 1.13    
    
    mask = torch.ones((batch_size, num_obj_points, 1), dtype=dtype, device=device)
    if quat:
        goal_feature_obj = decompose_pose_tensor(transform, cat=True)
    else:
        goal_feature_obj = transform[:, :3, :].reshape(batch_size, -1)
    goal_feature_obj = goal_feature_obj.unsqueeze(1).repeat(1, num_obj_points, 1)
    object_feature = torch.cat([mask, goal_feature_obj], axis=-1)
    
    # Backround points
    background_pcd_points = obs['background_pcd_points']
    if quat:
        goal_feature_bg = torch.zeros(7, device=device, dtype=dtype)
        goal_feature_bg[-1] = 1.
    else:
        goal_feature_bg = torch.eye(4, device=device, dtype=dtype)[:3, :].flatten()
    goal_feature_bg = goal_feature_bg.repeat(batch_size, num_bg_points, 1)
    # goal_feature_bg = decompose_pose_tensor(torch.eye(4, device=full_points.device).unsqueeze(0), cat=True)[0]
    bg_mask = torch.zeros((background_pcd_points.shape[0], background_pcd_points.shape[1], 1), dtype=dtype, device=device)
    background_feature = torch.cat([bg_mask, goal_feature_bg], axis=-1)
                        
    data_pos = torch.cat([object_pcd_points, background_pcd_points], axis=1)
    data_pos = data_pos.reshape(-1, 3)
    data_x = torch.cat([object_feature, background_feature], axis=1)
    data_x = data_x.reshape(-1, data_x.shape[-1])
    data_batch = torch.arange(object_pcd_points.shape[0], device=device).repeat_interleave(object_pcd_points.shape[1] + background_pcd_points.shape[1])
    return data_x, data_pos, data_batch
