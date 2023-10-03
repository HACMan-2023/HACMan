import numpy as np
import gym

from hacman.utils.transformations import transform_point_cloud, sample_idx

class RandomLocation(object):
    def __init__(self) -> None:
        return
    
    def get_action(self, obs):
        # Choose an index from the observation point cloud
        poke_idx = np.random.randint(len(obs['object_pcd_points']))
        return {'poke_idx': poke_idx,
                'action_location_score': np.zeros(len(obs['object_pcd_points']))}

"""
Abstracted Gym wrapper as an interface for HACMAN
"""
class BaseEnv(gym.Env):
    def __init__(self,
            object_pcd_size=400,
            background_pcd_size=400, **kwargs):
        
        self.spec = None
        self.metadata = {}

        # Observation space
        self.object_pcd_size = object_pcd_size
        self.background_pcd_size = background_pcd_size
        self.observation_space = gym.spaces.Dict(
                spaces={
                    "poke_idx": gym.spaces.Box(-np.inf, np.inf, (1,)),
                    "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    # "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                    "object_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                    "action_location_score": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size,)),
                    "background_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
                }
            )
        
        # Action space
        self.action_space = gym.spaces.Box(-1, 1, (3,))

        # Previous observation
        self.prev_obs = None

        # Action location policy
        self.random_location_policy = RandomLocation()

        super().__init__()
    

    def reset(self):
        obs = self.get_observations()
        self.prev_obs = obs
        return obs
    
    def set_prev_obs(self, obs):
        self.prev_obs = obs
        return True
        
    def step(self, action):
        # Take the environment step
        # Use the previous observation to get the location
        points = self.prev_obs['object_pcd_points']
        idx = self.prev_obs['poke_idx']
        location = points[idx]

        # TODO: Given the robot-object contact location "location" and the 
        # continuous paramter "action", Execute the action in simulation.
        pass
        
        # Compute the step outcome
        obs = self.get_observations()
        self.prev_obs = obs

        success, reward = self._evaluate_goal(obs, obs['goal_pose'])

        info = {"is_success": success,
                "action_param": action,
                "action_location": location,}

        done = False

        return obs, reward, done, info
        

    def get_observations(self):
        obs = {}
        # TODO: Change the follow placeholders
        obs['object_pose'] = np.eye(4)
        obs['goal_pose'] = np.eye(4)
        obs['object_pcd_points'] = np.zeros((self.object_pcd_size, 3))
        obs['background_pcd_points'] = np.zeros((self.object_pcd_size, 3))
    
        obj_idx = np.arange(obs['object_pcd_points'].shape[0])
        points = obs['object_pcd_points'][obj_idx]
        
        # Sample points to a fixed length
        obj_idx = sample_idx(len(points), self.object_pcd_size)
        obs['object_pcd_points'] = points[obj_idx, :]
        bg_idx = sample_idx(len(obs['background_pcd_points']), self.background_pcd_size)
        obs['background_pcd_points'] = obs['background_pcd_points'][bg_idx, :]
        
        # Choose location here:
        # Use a random location by default. Will be overwritten by the wrapper.
        location_info = self.random_location_policy.get_action(obs)
        obs.update(location_info)
        
        return obs
    
    def _evaluate_goal(self, obs, goal):
        Warning("This is a placeholder function. Please implement your own.")
        # Compute the flow
        old_pcd = obs['object_pcd_points']
        new_pcd = transform_point_cloud(obs['object_pose'], goal, old_pcd)
        flow = np.linalg.norm(new_pcd - old_pcd, axis=1)
        mean_flow = np.mean(flow)

        reward = -mean_flow
        success = mean_flow < 0.03

        return success, reward
    
if __name__ == "__main__":
    env = BaseEnv()
    env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0.1, 0.2, 0.3]))