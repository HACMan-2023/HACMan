"""
Should different action modes be included here?
"""

import os, sys

import numpy as np
import imageio
import pickle
import gym

from .base_env import BaseEnv, sample_idx
from hacman.utils.transformations import to_pose_mat, transform_point_cloud, decompose_pose_mat

from hacman.utils.plotly_utils import plot_pcd, plot_action, plot_pcd_with_score
from bin_env.util import angle_diff
from bin_env.poke_env import PokeEnv
# import plotly.graph_objects as go
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from hacman.algos.location_policy import RandomLocation


class HACManBinEnv(BaseEnv):
    def __init__(self, 
                 action_mode="per_point_action",
                 reward_scale=1.,
                 reward_mode="flow",
                 real_robot=False,
                 action_repeat=10,
                 fixed_ep_len=False,
                 reward_gripper_distance=None,
                 success_threshold=0.03,
                 **kwargs):
        super().__init__(
            object_pcd_size=400,
            background_pcd_size=400)
        
        self.real_robot = real_robot
        if self.real_robot:
            from franka_env_polymetis import FrankaPokeEnv
            self.env = FrankaPokeEnv(**kwargs)
        else:
            if action_mode == 'regress_action_only':
                # Only use position limits when the action continues from previous position
                kwargs['ignore_position_limits'] = False
            self.env = PokeEnv(**kwargs)
        
        self.action_repeat = action_repeat
        self.fixed_ep_len = fixed_ep_len
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.reward_gripper_distance = reward_gripper_distance
        self.success_threshold = success_threshold

        # Override the base env's observation and action space
        self.action_mode = action_mode
        if self.action_mode == "per_point_action":
            self.observation_space = gym.spaces.Dict(
                spaces={
                    "poke_idx": gym.spaces.Box(-np.inf, np.inf, (1,)),
                    "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "object_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                    "action_location_score": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size,)),
                    "background_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
                }
            )
            self.action_space = gym.spaces.Box(-1, 1, (3,))
            
        elif self.action_mode == "regress_location_and_action":
            self.observation_space = gym.spaces.Dict(
                spaces={
                    "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "object_size": gym.spaces.Box(-np.inf, np.inf, (3,)),
                    "object_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                    "background_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
                }
            )
            self.action_space = gym.spaces.Box(-1, 1, (6,))
        
        elif self.action_mode == "regress_action_only":
            self.observation_space = gym.spaces.Dict(
                spaces={
                    "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "object_size": gym.spaces.Box(-np.inf, np.inf, (3,)),
                    "object_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                    "background_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
                }
            )
            self.action_space = gym.spaces.Box(-1, 1, (3,))
        
        else:
            raise ValueError(f"Unknown action mode {self.action_mode}.")
        
                       
        self.goal = None
        
        self.step_count = 0
        self.missing_object_count = 0
        self.motion_failure_count = 0

    
    
    def reset(self, **kwargs):
        if self.real_robot:
            for i in range(10):
                raw_obs = self.env.reset(**kwargs)
                obs = self.process_observation(raw_obs)
                success, reward = self._evaluate_goal(obs, obs['goal_pose'])
                if success:
                    print(f'Resample goal because t=0 is at the goal: Attempt {i}')
                else:
                    break
            if success:
                print(f'Cannot find a goal within 10 attempts.')
                import ipdb; ipdb.set_trace()
        else:
            if self.action_mode == "regress_action_only":
                raw_obs = self.env.reset(
                    **kwargs, start_gripper_above_obj=True)
            else:
                raw_obs = self.env.reset(**kwargs)
                
            self.goal = self.env.sample_goal(raw_obs)
            self.env.set_goal(self.goal)
            obs = self.process_observation(raw_obs)
                
        self.prev_obs = obs
        return obs
    
    def step(self, action):
        start_location = None
        if self.action_mode == "per_point_action":
            # Per point regress action
            points = self.prev_obs['object_pcd_points']
            normals = self.prev_obs['object_pcd_normals']
            idx = self.prev_obs['poke_idx']
            location = points[idx]
            normal = normals[idx]
            
        elif self.action_mode == "regress_location_and_action":
            # Global regress location and action
            assert len(action) == 6
            
            # Raw action is in range [-1, 1]. We need to map the location to
            # the range of the bin.
            location = action[3:6]
            
            obj_pos, obj_ori = decompose_pose_mat(self.prev_obs['object_pose'], wxyz=True)
            
            cube_size = np.max(self.env.get_cube_size())
            bbox_half_size = cube_size / 2.0 + 0.02

            bbox_center = np.copy(obj_pos)
            bbox_center[2] = bbox_half_size + self.env.table_offset[2]

            location = location * bbox_half_size + bbox_center
            
            start_location = np.copy(location)
            start_location[2] = 2 * bbox_half_size + self.env.table_offset[2]
            # location += offset
            
            action = action[0:3]
            action[2] = - abs(action[2])
            normal = None
            
        elif self.action_mode == "regress_action_only":
            # Global regress action (continue from previous location)
            assert len(action) == 3
            action[2] = - abs(action[2])
            location = None
            normal = None
        
        else:
            raise ValueError(f"Unknown action mode {self.action_mode}.")
        
        raw_obs, _, _, poke_info = self.env.poke(
            location, normal, action, action_repeat=self.action_repeat,
            start_location=start_location)

        # Calculate step outcome
        self.step_count += 1
        if poke_info['poke_success'] == False:
            if poke_info['box_in_the_view'] == False:
                self.missing_object_count += 1
                print(f"{self.missing_object_count/self.step_count*100:.2f}% missing object within "
                    f"{self.step_count} total steps")
            else:
                self.motion_failure_count += 1
                if self.motion_failure_count % 100 == 0:
                    print(f"{self.motion_failure_count/self.step_count*100:.2f}% motion failure within "
                        f"{self.step_count} total steps")
        obs = self.process_observation(raw_obs)
        self.prev_obs = obs
            
        success, reward = self._evaluate_goal(obs, obs['goal_pose'])
        
        info = {"is_success": success,
                "action_param": action,
                "action_location": location,
                "poke_success":poke_info['poke_success'],
                "box_in_the_view": poke_info['box_in_the_view'],
                "object_name": poke_info['object_name'],
                }
                
        if "cam_frames" in poke_info.keys():
            info["cam_frames"] = poke_info["cam_frames"]
            
        if "fitness" in poke_info.keys():
            info["fitness"] = poke_info["fitness"]
                
        done = False
        if not self.fixed_ep_len:
            done = success
            
        return obs, reward, done, info
    
    def process_observation(self, obs):
        # Process the raw observation from robosuit
        new_obs = {}
        new_obs['object_pose'] = to_pose_mat(obs['cube_pos'], obs['cube_quat'])
        if 'goal_pose' in obs.keys():
            new_obs['goal_pose'] = obs['goal_pose']
        else:
            new_obs['goal_pose'] = self.goal.copy()

        if 'object_size' in obs.keys():
            new_obs['object_size'] = obs['object_size']

        if 'object_size' in obs.keys():
            new_obs['object_size'] = obs['object_size']

        # Add the gripper pose
        ee_pos, ee_quat = obs['robot0_eef_pos'], obs['robot0_eef_quat']
        new_obs['gripper_pose'] = to_pose_mat(ee_pos, ee_quat)

        if not 'object_pcd_points' in obs.keys() or obs['object_pcd_points'] is None:
            new_obs['object_pcd_points'] = None
            new_obs['object_pcd_normals'] = None
            new_obs['background_pcd_points'] = None     
            new_obs['poke_idx'] = None
            return new_obs

        obj_idx = np.arange(obs['object_pcd_points'].shape[0])
        points = obs['object_pcd_points'][obj_idx]
        normals = obs['object_pcd_normals'][obj_idx]
        
        # Sample points to a fixed length
        obj_idx = sample_idx(len(points), self.object_pcd_size)
        new_obs['object_pcd_points'] = points[obj_idx, :]
        new_obs['object_pcd_normals'] = normals[obj_idx, :]
        bg_idx = sample_idx(len(obs['background_pcd_points']), self.background_pcd_size)
        new_obs['background_pcd_points'] = obs['background_pcd_points'][bg_idx, :]
        
        if self.action_mode == "per_point_action":
            # Us a random policy by default. Will be overwritten by the wrapper.
            location_info = self.random_location_policy.get_action(new_obs)
            new_obs.update(location_info)

        if self.real_robot:
            # Save goal frames
            new_obs['raw_goal_frames'] = obs['goal_frames']
            
        return new_obs
    
    def _evaluate_goal(self, obs, goal):
        if obs['object_pcd_points'] is None:
            reward = 0
            success = False
            return success, reward
    
        if self.reward_mode == "flow":
            old_pcd = obs['object_pcd_points']
            new_pcd = transform_point_cloud(obs['object_pose'], goal, old_pcd)
            flow = np.linalg.norm(new_pcd - old_pcd, axis=-1)
            mean_flow = np.mean(flow, axis=-1)
            reward = -mean_flow

            # Add a distance term
            if self.action_mode in ["regress_action_only", "regress_location_and_action"] and (self.reward_gripper_distance is not None):
                gripper_pos, _ = decompose_pose_mat(obs['gripper_pose'])
                dists = np.linalg.norm(old_pcd - gripper_pos, axis=1)
                min_dist = np.min(dists)
                reward -= max(min_dist - 0.05, 0.0) * self.reward_gripper_distance
            
            success = mean_flow < self.success_threshold
        
        elif self.reward_mode == "pose_diff":
            DeprecationWarning("Pose diff reward is deprecated.")
            object_pos, object_ori = decompose_pose_mat(obs['object_pose'])
            goal_pos, goal_ori = decompose_pose_mat(goal)
            pos_diff = np.linalg.norm(object_pos - goal_pos)
            ori_diff = angle_diff(object_ori, goal_ori) / np.pi * 180.0
            reward = - (50 * pos_diff + 0.2 * ori_diff)
            success = - reward < self.success_threshold
        
        else:
            raise NotImplementedError

        return success, reward
