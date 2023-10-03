import numpy as np
from .base_env import BaseEnv, RandomLocation
from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx
import gym

"""
Abstracted Gym wrapper as an interface for HACMAN
"""
class SimpleEnv(BaseEnv):
    def __init__(self,
            object_pcd_size=400,
            background_pcd_size=400, **kwargs):
        self.sim = SimpleCubeSim()
        super().__init__(object_pcd_size, background_pcd_size, **kwargs)
    
        self.reset()

    def reset(self):
        self.sim.reset()

        # Get the goal pose
        goal_pos = self.sim.get_goal_pos()
        self.goal = self._cube_pos_to_pose_mat(goal_pos)
        
        obs = self.get_observations()
        self.prev_obs = obs
        return obs
        

    def step(self, motion):
        # Take the environment step
        # Use the previous observation to get the location
        points = self.prev_obs['object_pcd_points']
        idx = self.prev_obs['poke_idx']
        location = points[idx.item()]

        # Given the robot-object contact location "location" and the 
        # continuous paramter "motion", Execute the action in simulation.
        self.sim.poke(location, motion)
        
        # Compute the step outcome
        obs = self.get_observations()
        self.prev_obs = obs

        success, reward = self._evaluate_goal(obs, obs['goal_pose'])

        info = {"is_success": success,
                "action_param": motion,
                "action_location": location,}

        done = False

        return obs, reward, done, info
        

    def get_observations(self):
        obs = {}

        obs['object_pose'] = self._cube_pos_to_pose_mat(self.sim.get_cube_pos())
        obs['goal_pose'] = np.copy(self.goal)

        # Point cloud segementation
        points, segmentation = self.sim.get_point_cloud()
        object_points = points[segmentation == 1]        # Cube points
        background_points = points[segmentation == 0]    # Table points
        
        # Sample points to a fixed length
        obj_idx = sample_idx(len(object_points), self.object_pcd_size)
        obs['object_pcd_points'] = object_points[obj_idx, :]
        bg_idx = sample_idx(len(background_points), self.background_pcd_size)
        obs['background_pcd_points'] = background_points[bg_idx, :]
        
        # Choose location here:
        # Use a random location by default. Will be overwritten by the wrapper.
        location_info = self.random_location_policy.get_action(obs)
        obs.update(location_info)
        
        return obs


    def get_normalize_param(self):
        offset = np.zeros(3)
        scale = np.ones(3) * self.sim.table_size
        return offset, scale


    def _evaluate_goal(self, obs, goal):
        # Compute the flow
        old_pcd = obs['object_pcd_points']
        new_pcd = transform_point_cloud(obs['object_pose'], goal, old_pcd)
        flow = np.linalg.norm(new_pcd - old_pcd, axis=1)
        mean_flow = np.mean(flow)

        reward = -mean_flow
        success = mean_flow < 0.05

        return success, reward
    
    def _cube_pos_to_pose_mat(self, pos):
        pos, quat = np.array([*pos, 0]), np.array([0, 0, 0, 1])
        pose_mat = to_pose_mat(pos, quat, input_wxyz=False)
        return pose_mat


class SimpleCubeSim(object):
    """A simple cube simulation for poking and interacting with a cube on a table.

    The simulation initializes a cube with a given size and table size. It provides
    methods for resetting the cube state, poking the cube at a specific location,
    obtaining the goal pose (which is always at the center of the table), retrieving
    the current pose of the cube, and generating a point cloud representing the table
    and the translated cube.

    **See scripts/test_simple_env.py for visualization.**

    Usage:
        sim = SimpleCubeSim()          # Create a SimpleCubeSim object
        sim.reset()                    # Reset the cube state
        sim.poke(location, motion)     # Poke the cube at a specific location with motion
        goal_pose = sim.get_goal_pose()      # Get the goal pose
        object_pose = sim.get_object_pose()  # Get the current pose of the cube
        point_cloud, segmentation = sim.get_point_cloud()
                                    # Get the point cloud representing
                                    the table and the cube, along with the segmentation mask

    Attributes:
        cube_size (float): Half size of the cube.
        table_size (float): Half size of the table.
        table_points (ndarray): Points representing the table in the point cloud.
        cube_points (ndarray): Points representing the cube in the point cloud.
    """
    def __init__(self):
        self.cube_size = 0.1    # cube half size
        self.table_size= 1.0    # table half size

        # For point cloud synthesis
        self.table_points = self._render_table_points(res=0.02)
        self.cube_points = self._render_cube_points(res=0.02)

        self.reset()   # Initialize the cube state
    
    def reset(self, state=None):
        if state is not None:
            self.cube_state = np.copy(state)
        else:
            # self.cube_state = np.random.uniform(
            #     -self.table_size + self.cube_size, self.table_size - self.cube_size, size=2)
            self.cube_state = np.array([0.5, 0.5])
    
    def poke(self, location, motion):
        """
        Simplified poke simulation.

        A force is applied to the cube only when it is pushing
        on a side face. Pulling and poking on the top face does not do anything.

        Args:
            location: 3D location of the poke on the cube
            motion: 3D motion of the poke (delta translation, x, y, z)
        """
        
        if np.isclose(location[2], 2 * self.cube_size): # Top face
            translation = np.zeros(2)
        else:
            normal = self._get_normal(location)
            if np.dot(normal, motion) < 0:
                translation = motion[:2]
            else:
                translation = np.zeros(2)

        # Update the cube state
        self.cube_state[:2] += translation
    
    def get_goal_pos(self):
        goal = np.zeros(2)  # It is always at the center of the table
        return goal

    def get_cube_pos(self):
        return self.cube_state

    def get_point_cloud(self):
        # Synthesize the point cloud by combining the table and the translated cube points
        cube_x, cube_y = self.cube_state
        cube_points = self.cube_points + np.array([cube_x, cube_y, 0.0])
        points = np.concatenate([self.table_points, cube_points], axis=0)

        # Segmentation mask
        segmentation = np.ones(len(points), dtype=np.int32)
        segmentation[:len(self.table_points)] = 0
        return points, segmentation

    def _get_normal(self, location):
        cube_com = np.array([self.cube_state[0], self.cube_state[1], self.cube_size/2.0])
        relative_location = (location - cube_com) / self.cube_size
        
        face_direction_mask = np.isclose(np.abs(relative_location), 1.0)  # Determine if the face faces x, y, or z
        normal = relative_location * face_direction_mask
        return normal
    
    def _render_table_points(self, res=0.02):
        points = []
        for x in np.arange(-self.table_size, self.table_size, res):
            for y in np.arange(-self.table_size, self.table_size, res):
                points.append([x, y, 0.0])
        return np.array(points)

    def _render_cube_points(self, res=0.02):
        points = []
        # Cube sides
        for z in np.arange(0, 2 * self.cube_size, res):
            for x in np.arange(-self.cube_size, self.cube_size + 1e-3, res):
                points.append([x, -self.cube_size, z])
                points.append([x, self.cube_size, z])
            for y in np.arange(-self.cube_size + 1e-3, self.cube_size, res):
                points.append([-self.cube_size, y, z])
                points.append([self.cube_size, y, z])
        # Cube top
        for x in np.arange(-self.cube_size, self.cube_size + 1e-3, res):
            for y in np.arange(-self.cube_size, self.cube_size + 1e-3, res):
                points.append([x, y, 2 * self.cube_size])

        # Add a mask digit to the points
        return np.array(points)
