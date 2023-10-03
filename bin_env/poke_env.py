import os, sys
import pickle
import numpy as np
import robosuite
import copy
from bin_env.base_env import BaseEnv
import open3d as o3d
from bin_env.utils.point_cloud_utils import convert_depth, get_point_cloud, add_additive_noise_to_xyz, dropout_random_ellipses
from robosuite.utils.control_utils import orientation_error
from bin_env.utils.transformations import to_pose_mat, decompose_pose_mat
from robosuite.controllers import load_controller_config

VERBOSE = False
robosuite.utils.macros.SIMULATION_TIMESTEP = 0.002

class PokeEnv(BaseEnv):
    def __init__(
            self,
            # Specific to PokeEnv
            ik_precontact=True,
            transparent_gripper_base=True,
            planar_action=False,
            record_from_cam=None,
            resting_time=2, 
            goal_mode='fixed',
            controller="osc_no_ori.json",  # "OSC_POSE"
            ignore_position_limits=True,
            renderer='offscreen',
            # Will be passed to BaseEnv
            robots='Virtual',
            close_gripper=True,
            object_dataset='cube',
            object_name=None,
            control_freq=2,
            ignore_done=True,
            render_camera='agentview',
            location_noise=0.,
            pcd_noise=None,
            object_voxel_down_sample=0.01,
            friction_config='default',
            **kwargs
    ):

        """
        Setup parameters for PokeEnv
        """
        self.planar_action = planar_action
        self.ik_precontact = ik_precontact
        self.transparent_gripper_base = transparent_gripper_base
        self.record_from_cam = record_from_cam 
        self.cam_frames = []
        self.goal = None
        self.resting_time = resting_time
        self.location_noise = location_noise
        self.pcd_noise = pcd_noise # [2, 30, 2, 0.005]
        self.object_voxel_down_sample = object_voxel_down_sample
        
        current_folder = os.path.dirname(os.path.abspath(__file__))
        
        # Setup goal sampling
        self.goal_mode = goal_mode
        self.goal_list = None
        if object_dataset == 'cube':
            if goal_mode == 'any-old':
                pose_file = os.path.join(current_folder, 'data/Dataset0004/states.npy')
            elif goal_mode == 'any-old-eval':
                pose_file = os.path.join(current_folder, 'data/Dataset0004_eval/states.npy')
            elif goal_mode == 'oop-old':
                pose_file = os.path.join(current_folder, 'data/Dataset0004/states_out_of_plane.npy')
            elif goal_mode == 'oop-old-eval':
                pose_file = os.path.join(current_folder, 'data/Dataset0004_eval/states_out_of_plane.npy')
            elif goal_mode in ['fixed', 'translation', 'upright', 'fixed_off_center', 'off_center']:
                pose_file = None
            else:
                raise ValueError("Unknown goal_mode: {}".format(goal_mode))
            
            if pose_file is not None:
                self.goal_list = {'cube': {'poses':np.load(pose_file)}}
        
        elif object_dataset in {'housekeep', 'housekeep_all'}:
            if goal_mode == 'any':
                if 'object_scale_range' in kwargs.keys() and kwargs['object_scale_range'] is not None:
                    assert kwargs['object_scale_range'][0] == kwargs['object_scale_range'][1], "object_scale_range should be a single value!"
                pose_file = os.path.join(current_folder, f'data/{object_dataset}/poses_fixed_size.pk')

            elif goal_mode == 'any_var_size':
                pose_file = os.path.join(current_folder, f'data/{object_dataset}/poses_variable_size_v1.pk')
            
            elif goal_mode == 'any_var_size_with_wall':
                pose_file = os.path.join(current_folder, f'data/{object_dataset}/poses_variable_size_with_wall.pk')

            elif goal_mode in ['fixed', 'translation', 'upright', 'fixed_off_center', 'off_center']:
                pose_file = None

            else:
                raise ValueError("Unknown goal_mode: {}".format(goal_mode))
            
            if pose_file is not None:
                assert os.path.exists(pose_file), "Pose file does not exist: {}".format(pose_file)
                with open(pose_file, 'rb') as f:
                    self.goal_list = pickle.load(f)
        
        else:
            ValueError("Unknown object_dataset: {}".format('object_dataset'))

        """
        Setup configurations for BaseEnv
        """
        if robots == 'Virtual':
            kwargs['initial_qpos'] = np.array([0.45, 0., 0.4, 0., np.pi, np.pi/2])
        else:
            kwargs['initial_qpos'] = np.array([0, -0.2743, 0, -2.263, 0, 2.008, -0.7946])
        kwargs['initialization_noise'] = None
        
        # if 'object_dataset' in kwargs and 'housekeep' in kwargs['object_dataset']:
        # kwargs['table_friction'] = (0., 0.3, 0.1)
        if friction_config == 'default':
            kwargs['table_friction']=(0.5, 0.005, 0.0001)
        elif friction_config == 'low':
            kwargs['table_friction']=(0.3, 0.005, 0.0001)
        elif friction_config == 'high':
            kwargs['table_friction']=(0.95, 0.05, 0.001)
        elif friction_config == 'robosuite':
            kwargs['table_friction']=(0.95, 0.3, 0.1)
        else:
            raise NotImplementedError    
        
        kwargs['table_solref'] = (0.01, 1)
        kwargs['table_xml'] = 'bin_arena_all_side_planes.xml'
        kwargs['hard_reset'] = object_name is None and object_dataset != 'cube'

        # Setup Controller
        if '.json' in controller:
            controller_configs = load_controller_config(custom_fpath=os.path.join(current_folder, 'controller_config', controller))
        else:
            controller_configs = load_controller_config(default_controller='OSC_POSE')
            
        if ignore_position_limits:
            controller_configs['position_limits'] = None
        kwargs.update(controller_configs=controller_configs)
        
        # Setup Renderer
        if renderer == 'onscreen':
            # Use onscreen renderer
            render_config = dict(has_renderer=True,
                                has_offscreen_renderer=False,
                                use_camera_obs=False)
            kwargs.update(render_config)
        elif renderer == 'offscreen':
            # Use offscreen renderer
            render_config = dict(has_renderer=False,
                                has_offscreen_renderer=True,
                                use_camera_obs=True,
                                camera_names=['leftview', 'rightview', 'agentview'],
                                camera_depths=True,
                                camera_segmentations='instance'
                                )
            kwargs.update(render_config)

        super().__init__(robots=robots,
            close_gripper=close_gripper,
            object_dataset=object_dataset,
            object_name=object_name,
            control_freq=control_freq,
            ignore_done=ignore_done,
            render_camera=render_camera,
            **kwargs)
                
        # Action location range (after BaseEnv init)
        self.location_scale = None
        self.location_center = None
        return
    
    def _reset_internal(self):
        super()._reset_internal()
        
        # Update location scale
        self.location_scale = self.table_size_curr/2 + np.array([-0.03, -0.03, 0])
        # make the range smaller for x and y, but not for z
        self.location_center = self.table_offset + np.array([0, 0, self.table_size_curr[2]/2])
        # self.table_offset is at the center of the bottom of the bin.
        # self.location_center is at the center of the entire free space of the bin.
        
        if self.transparent_gripper_base:
            gid = self.sim.model.geom_name2id('gripper0_hand_collision')
            self.sim.model.geom_conaffinity[gid] = 0
            self.sim.model.geom_contype[gid] = 0    
            gid = self.sim.model.geom_name2id('gripper0_hand_visual')
            self.sim.model.geom_rgba[gid] = np.array([1, 1, 1, 0.3])
        
        if self.robots[0].gripper_type == 'default':
            eef_body_id = self.sim.model.body_name2id('gripper0_eef')
            if self.close_gripper:
                self.sim.model.body_pos[eef_body_id] = np.array([0, 0, 0.105])
            else:
                self.sim.model.body_pos[eef_body_id] = np.array([-0.053, 0, 0.105])
        
        # Hide global coordinate markers
        for site_name in ['xaxis', 'yaxis', 'zaxis']:
            sid = self.sim.model.site_name2id(site_name)
            self.sim.model.site_rgba[sid] = np.array([1, 1, 1, 0])

        return
    
    def set_goal(self, goal):
        # Convert matrix into vector
        self.goal = decompose_pose_mat(goal, cat=True, wxyz=True)
        self.maybe_render()
        return
    
    def show_goal(self):
        goal = self.goal
        if goal is not None:
            body_id = self.sim.model.body_name2id(self.cube_target.root_body)
            self.sim.model.body_pos[body_id] = goal[:3]
            self.sim.model.body_quat[body_id] = goal[3:]
            self.sim.forward()
        return
    
    def hide_goal(self):
        body_id = self.sim.model.body_name2id(self.cube_target.root_body)
        self.sim.model.body_pos[body_id][2] = -0.2
        self.sim.forward()
        return

    def reset(self, object_pose=None, object_size=None, goal=None, hard_reset=None, object_name=None, start_gripper_above_obj=False,
              attempt=0):
        if object_size is not None:
            self.object_size_x_min = object_size[0]
            self.object_size_x_max = object_size[0]
            self.object_size_y_min = object_size[1]
            self.object_size_y_max = object_size[1]
            self.object_size_z_min = object_size[2]
            self.object_size_z_max = object_size[2]
        
        if object_name is not None:
            self.object_sampler.set_object(object_name)

        previous_hard_reset = self.hard_reset
        # Override hard_reset if specified
        self.hard_reset = hard_reset if hard_reset is not None else self.hard_reset
        super().reset()
        self.hard_reset = previous_hard_reset
        
        self.goal = goal
        # Update mujoco visual of the goal
        if self.goal is not None:
            body_id = self.sim.model.body_name2id(self.cube_target.root_body)
            self.sim.model.body_pos[body_id] = self.goal[:3]
            self.sim.model.body_quat[body_id] = self.goal[3:]
            self.sim.forward()
        else:
            body_id = self.sim.model.body_name2id(self.cube_target.root_body)
            self.sim.model.body_pos[body_id] = np.array([0,0,-1]) # hide it
            self.sim.forward()
                
        if object_pose is not None:
            self.sim.data.set_joint_qpos(self.cube.joints[0], object_pose)
            self.sim.forward()
            self.robots[0].controller.run_controller()
        
        if start_gripper_above_obj:
            obj_size = np.max(self.get_cube_size())
            
            gripper_pos = self.sim.data.get_joint_qpos(self.cube.joints[0])[:3]
            gripper_pos[2] = obj_size + self.table_offset[2] + 0.05
            self.move_to(gripper_pos)

        # Run simulation to rest the object.
        self.run_simulation(total_execution_time=self.resting_time)

        self.visualize(vis_settings=dict(env=False, grippers=True, robots=False))
        obs = self._get_observations(force_update=True)
        
        if 'object_pcd_points' in obs.keys() and obs['object_pcd_points'] is None:
            assert attempt < 5, "Failed to get object point cloud after 5 attempts."
            print(f'no object point cloud during reset. try to reset again. attempt {attempt}')
            obs = self.reset(hard_reset=False, attempt=attempt+1)

        return obs

    def get_cube_size(self):
        if hasattr(self.cube, 'scaled_size'):
            cube_size = self.cube.scaled_size
        elif hasattr(self.cube, 'size'):
            cube_size = np.array(self.cube.size) * 2
        else:
            raise ValueError("Unknown cube size.")
        
        return cube_size

    """
    Movement related
    """
    def sample_random_poke(self, obs=None, points=None, normals=None):
        # Take a dictionary of obs as input, or directly take points and normals as input
        if obs is not None:
            assert points is None and normals is None
            points = obs['object_pcd_points']
            normals = obs['object_pcd_normals']
        idx = np.random.randint(len(points))
        location = points[idx]
        normal = normals[idx]
        action = np.random.rand(self.action_dim)*2-1
        if self.planar_action:
            action[-1] = 0
        if action.dot(normal) > 0:
            action *= -1
        return location, normal, action, idx

    def poke(self, location, normal, action, action_repeat=10, start_location=None):
        '''
        Set normal to None to directly move to locations.
        '''
        self.cam_frames = []
        box_in_the_view = True
        # Save original object pose
        obj_pose = self.sim.data.get_joint_qpos(self.cube.joints[0]).copy()

        if location is not None and normal is None:
            assert start_location is not None, "Need to provide start location in regress_location_and_action mode."
        
        try:
            # Contiguous movements
            if location is None:
                pass    # No additional movement for contiguous movements

            # Regress locations and actions
            elif normal is None:
                # assert self.collision_check(start_location)
                self.robots[0].reset(deterministic=True)
                self.sim.forward()

                # Approach from start location (should be top)
                assert self.move_to_from_top(location=location, start_location=start_location)

            # Per-point actions
            else:
                # assert self.collision_check(location)
                self.robots[0].reset(deterministic=True)
                self.sim.forward()

                noise = (np.random.rand(3)*2-1)*self.location_noise
                location += noise
                assert self.move_to_contact(location=location, normal=normal)

        except:
            success = False

        else:
            # Execute action parameters
            success = True
            for _ in range(action_repeat):
                self.run_simulation(action)
        
        # Reset robot if using location
        if location is not None:
            self.robots[0].reset(deterministic=True)
            self.sim.forward()
            self.run_simulation(total_execution_time=self.resting_time)
        
        obs = self._get_observations(force_update=True)        
        if 'object_pcd_points' in obs.keys():
            if obs['object_pcd_points'] is None:
                box_in_the_view = False
                print(f'no object point cloud. try to reset.')
                obs = self.reset(hard_reset=False)
        else:
            obs['object_pcd_points'] = None

        # Add object name to the info
        if hasattr(self.cube, 'mesh_name'):
            obj_name = self.cube.mesh_name
        else:
            obj_name = "cube"

        info = {
            'poke_success': success and box_in_the_view,
            'box_in_the_view':box_in_the_view,
            'cam_frames': copy.copy(self.cam_frames),
            'object_name': obj_name}
        self.cam_frames.clear()
        # Following env.step() output format
        return obs, 0, False, info
    
    def collision_check(self, location):
        scaled_location = (location - self.location_center)/self.location_scale
        return np.all(np.abs(scaled_location) <= 1.01)

    def move_to_contact(self, location, normal):
        ee_pos = self.robots[0].controller.ee_pos
        precontact = location + normal * 0.02

        if self.robots[0].name == 'Virtual' and self.ik_precontact:
            # Set the floating gripper to the location directly (ik)
            preprecontact = location + normal * 0.04
            s1, s2 = self.move_to(pos=preprecontact, set_qpos=True)
            if not (s1 and s2):
                return False
            self.move_to(pos=precontact)
        else:
            # Gradually move to the contact point using the low-level controller
            midpoint = (precontact - ee_pos)*2/3 + ee_pos
            
            waypoints = 5
            for i in range(waypoints):
                new_pos = (midpoint - ee_pos)/waypoints*(i+1) + ee_pos
                s1, s2 = self.move_to(pos=new_pos)
                if not (s1 and s2):
                    return False
            
            waypoints = 5
            ee_pos = self.robots[0].controller.ee_pos
            for i in range(waypoints):
                new_pos = (precontact - ee_pos)/waypoints*(i+1) + ee_pos
                s1, s2 = self.move_to(pos=new_pos)
                if not (s1 and s2):
                    return False

        s1, s2 = self.move_to(pos=location, suppress_collision_warning=True) # it's ok if the object moves a little during contact
        return s1

    def move_to(self, pos, quat=None, set_qpos=False, run_simulation_time=None,
                suppress_collision_warning=False):
        # Designed to move in free space

        # Save original object pose
        obj_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        obj_mat = np.array(self.sim.data.body_xmat[self.cube_body_id].reshape([3, 3]))

        # Visualize target eef pose
        target_eef_id = self.sim.model.body_name2id('target_eef')
        self.sim.model.body_pos[target_eef_id] = pos
        if quat is not None:
            self.sim.model.body_quat[target_eef_id] = quat
        self.sim.forward()
        
        site_id = self.sim.model.site_name2id('target_eef:grip_site')
        ee_pos = np.array(self.sim.data.site_xpos[site_id])
        ee_ori_mat = np.array(self.sim.data.site_xmat[site_id].reshape([3, 3]))
        
        if set_qpos:
            assert self.robots[0].name == 'Virtual'
            # Set the floating gripper to the location directly (ik)
            eef_body_id = self.sim.model.body_name2id('gripper0_eef')
            offset = self.sim.model.body_pos[eef_body_id]
            qpos = ee_pos + offset
            
            self.sim.data.set_joint_qpos('robot0_slide_joint0', qpos[0])
            self.sim.data.set_joint_qpos('robot0_slide_joint1', qpos[1])
            self.sim.data.set_joint_qpos('robot0_slide_joint2', qpos[2])
            
            self.sim.forward()
            self.sim.step()
            self.robots[0].controller.update(force=True)
            self.maybe_render()
        else:
            self.run_simulation(ee_pos=ee_pos, ee_ori_mat=ee_ori_mat,
                                total_execution_time=run_simulation_time)

        # Check if the gripper reaches the pose
        reachable = True
        ee_pos_after = self.robots[0].controller.ee_pos
        ee_ori_mat_after = self.robots[0].controller.ee_ori_mat.reshape(3,3)
        pos_diff = np.linalg.norm(ee_pos_after - ee_pos)
        ori_diff = np.abs(orientation_error(ee_ori_mat_after, ee_ori_mat)/np.pi*180)
        if pos_diff > 0.02 or np.any(ori_diff > 5):
            if VERBOSE:
                print(f"fail: cannot reach the ee pose.\t Current pose: {ee_pos_after} \t"
                      f"Desired Pose: {ee_pos}\t Diff:{pos_diff}\t{ori_diff}")
            reachable = False

        # Check if the object has moved
        collision_free = True
        obj_pos_after = np.array(self.sim.data.body_xpos[self.cube_body_id])
        obj_mat_after = np.array(self.sim.data.body_xmat[self.cube_body_id].reshape(3,3))
        pos_diff = np.linalg.norm(obj_pos_after - obj_pos)
        ori_diff = np.abs(orientation_error(obj_mat_after, obj_mat)/np.pi*180)
        if pos_diff > 0.02 or np.any(ori_diff > 5):
            if VERBOSE and not suppress_collision_warning:
                print(f"fail: object moved during initialization.\t {pos_diff}\t{ori_diff}")
            collision_free = False

        # Return success: the location is reachable and the object didn't move
        return reachable, collision_free
    

    def move_to_from_top(self, location, start_location=None):
        # First move to the start location
        s1, s2 = self.move_to(start_location, set_qpos=True)
        self.maybe_render()
        if not (s1 and s2):
            return False
        
        # Generating several waypoints to move the gripper from the top
        # Calls move_to to move between waypoints later
        site_id = self.sim.model.site_name2id('target_eef:grip_site')
        ee_pos = np.array(self.sim.data.site_xpos[site_id])
        ee_ori_mat = np.array(self.sim.data.site_xmat[site_id].reshape([3, 3]))
        ee_z = ee_pos[2]
        loc_z = location[2]

        # Create the waypoints
        assert np.isclose(ee_pos[:2], location[:2], atol=0.01).all() , "Warning: the gripper is not right above the target location."
        waypoints_z = np.arange(loc_z, ee_z, 0.02)[::-1]
        waypoints = [[location[0], location[1], z] for z in waypoints_z]
        
        # Move to each waypoint
        for waypoint in waypoints:
            reachable, collision_free = self.move_to(waypoint, set_qpos=False)
            self.maybe_render()

            if not reachable:
                break
        
        return True


    def run_simulation(self, action=None, ee_pos=None, ee_ori_mat=None, total_execution_time=None):
        # Run simulation without taking a env.step
        action = np.zeros(self.action_dim) if action is None else action
        total_execution_time = self.control_timestep if total_execution_time is None else total_execution_time
        for i in range(int(total_execution_time/self.control_timestep)):
            self.robots[0].controller.set_goal(action, set_pos=ee_pos, set_ori=ee_ori_mat)
            for _ in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                torques = self.robots[0].controller.run_controller()
                low, high = self.robots[0].torque_limits
                torques = np.clip(torques, low, high)
                if self.close_gripper:
                    gripper_action = 1
                else:
                    gripper_action = -1
                self.robots[0].grip_action(gripper=self.robots[0].gripper, gripper_action=[gripper_action])
                self.sim.data.ctrl[self.robots[0]._ref_joint_actuator_indexes] = torques
                self.sim.step()
            self.maybe_render()

    def maybe_render(self):
        if self.has_offscreen_renderer:
            if self.record_from_cam is not None:
                self.show_goal()
                img = self.sim.render(
                    camera_name=self.record_from_cam,
                    width=1280,
                    height=720)
                self.cam_frames.append(img)
                # import matplotlib.pyplot as plt
                # import time
                # plt.imsave(f'agent_{self.cube.mesh_name}_{int(time.time())}.png', img[-1::-1, :, :])
                self.hide_goal()
            
        elif self.has_renderer:
            # Always render intermediate steps if there is an onscreen renderer.
            # Turn this off by setting renderer to None.
            self.show_goal()
            super().render()
            self.hide_goal()
            

    """
    Point cloud related
    """
    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)

        if self.has_offscreen_renderer:
            pcd_dict = self.get_point_cloud(obs)

            # Try 5 times:
            if self.pcd_noise is not None and pcd_dict['object_pcd_points'] is None:
                for i in range(5):
                    print(f'resample pcd: {i}')
                    pcd_dict = self.get_point_cloud(obs)
                    if pcd_dict['object_pcd_points'] is not None:
                        break
            obs.update(pcd_dict)
        if self.goal is not None:
            obs['goal'] = self.goal

        # Add oject size to the observation
        if hasattr(self.cube, 'scaled_size'):
            obs['object_size'] = self.cube.scaled_size
        elif hasattr(self.cube, 'size'):
            obs['object_size'] = self.cube.size
        else:
            raise ValueError("No object size found.")
            
        return obs

    def get_point_cloud(self, obs):
        box_in_the_view = True
        pcd_dict = {}
        pc_list = []
        pc_color_list = []
        mask_list = []

        for cam in self.camera_names:
            color = obs[cam + '_image'][-1::-1]
            depth = obs[cam + '_depth'][-1::-1][:, :, 0]
            if self.pcd_noise is not None:
                depth = dropout_random_ellipses(depth_img=depth, dropout_mean=self.pcd_noise[0], 
                                                gamma_shape=self.pcd_noise[1], gamma_scale=self.pcd_noise[2])

            # Segmentation from MuJoCo
            seg = obs[cam + '_segmentation_instance'][-1::-1]
            obj_mask = (seg == 1).reshape(color.shape[0], color.shape[1])

            depth = convert_depth(self, depth)
            pc = get_point_cloud(self, depth, camera_name=cam)
            if self.pcd_noise is not None:
                pc = add_additive_noise_to_xyz(pc.reshape(color.shape), gaussian_scale_range=[0.0, self.pcd_noise[3]]).reshape(-1, 3)
            pc_color = color.reshape(-1, 3).astype(np.float64)/256
            
            pc_list.append(pc)
            pc_color_list.append(pc_color)
            mask_list.append(obj_mask.reshape(-1))

        pc = np.concatenate(pc_list)
        pc_color = np.concatenate(pc_color_list)
        pc_mask = np.concatenate(mask_list)
        
        # Remove unnecessary points outside of the bin
        scene_mask = self.within_arena(pc, margin=np.array([-0.01, -0.01, -0.1]), check_z=True)
        pc = pc[scene_mask]
        pc_color = pc_color[scene_mask]
        pc_mask = pc_mask[scene_mask]
        if pc_mask.sum() <= 10:
            # import matplotlib.pyplot as plt
            # import time
            # plt.imsave(f'agent_{self.cube.mesh_name}_{int(time.time())}.png', obs['agentview_image'][-1::-1, :, :])
            # plt.imsave(f'left_{self.cube.mesh_name}_{int(time.time())}.png', obs['leftview_image'][-1::-1, :, :])
            # plt.imsave(f'right_{self.cube.mesh_name}_{int(time.time())}.png', obs['rightview_image'][-1::-1, :, :])
            box_in_the_view = False
        
        if box_in_the_view:
            try:
                # Downsample object pcd
                object_pcd = o3d.geometry.PointCloud()
                object_pcd.points = o3d.utility.Vector3dVector(pc[pc_mask])
                object_pcd.colors = o3d.utility.Vector3dVector(pc_color[pc_mask])
                # print(len(object_pcd.points))
                # o3d.visualization.draw_geometries([object_pcd])
                object_pcd = object_pcd.voxel_down_sample(self.object_voxel_down_sample)
                object_pcd.estimate_normals()
                object_pcd.orient_normals_consistent_tangent_plane(10)
                # print(len(object_pcd.points))
                # o3d.visualization.draw_geometries([object_pcd])
            
            except:
                box_in_the_view = False
        
        # Downsample background pcd
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(pc[~pc_mask])
        bg_pcd.colors = o3d.utility.Vector3dVector(pc_color[~pc_mask])
        # print(len(bg_pcd.points))
        # o3d.visualization.draw_geometries([bg_pcd])
        bg_pcd = bg_pcd.voxel_down_sample(0.02)
        # print(len(bg_pcd.points))
        # o3d.visualization.draw_geometries([bg_pcd])
        
        if box_in_the_view:
            pcd_dict['object_pcd_points'] = np.asarray(object_pcd.points)
            pcd_dict['object_pcd_colors'] = np.asarray(object_pcd.colors)
            pcd_dict['object_pcd_normals'] = np.asarray(object_pcd.normals)
        else:
            pcd_dict['object_pcd_points'] = None
            pcd_dict['object_pcd_colors'] = None
            pcd_dict['object_pcd_normals'] = None
        pcd_dict['background_pcd_points'] = np.asarray(bg_pcd.points)
        pcd_dict['background_pcd_colors'] = np.asarray(bg_pcd.colors)
        return pcd_dict

    def within_arena(self, points, margin=None, check_z=False):
        # Check if the points are within the box on the table
        # Input: N x 3
        if margin is None:
            margin = np.zeros(3)
        x_min = self.table_offset[0] - self.table_size_curr[0]/2 + margin[0]
        x_max = self.table_offset[0] + self.table_size_curr[0]/2 - margin[0]
        y_min = self.table_offset[1] - self.table_size_curr[1]/2 + margin[1]
        y_max = self.table_offset[1] + self.table_size_curr[1]/2 - margin[1]
        valid = (points[:, 0] >= x_min) * (points[:, 0] <= x_max) * \
                (points[:, 1] >= y_min) * (points[:, 1] <= y_max)
        if check_z:
            z_min = self.table_offset[2] + margin[2]
            z_max = self.table_offset[2] + self.table_size_curr[2] - margin[2]
            valid *= (points[:, 2] >= z_min) * (points[:, 2] <= z_max)
        return valid
    
    """
    Others
    """
    def reward(self, action=None):
        return 0
    
    def sample_goal(self, obs):
        scale = None
        if self.goal_list is not None:
            if hasattr(self, 'cube') and hasattr(self.cube, 'mesh_name'):
                mesh_name = self.cube.mesh_name
                scale = self.cube.scale

                # Find the goals with the same scale
                goal_scales = self.goal_list[mesh_name]['scales']
                scale_mask = np.isclose(np.linalg.norm(goal_scales - scale, axis=1), 0, rtol=1e-2)
                if not np.any(scale_mask):
                    raise ValueError('No goals with the same scale!')
                goals = self.goal_list[mesh_name]['poses'][scale_mask]
            else:
                mesh_name = 'cube'
                goals = self.goal_list[mesh_name]['poses']
            
            goal_idx = np.random.choice(len(goals))
            goal = goals[goal_idx]
            
            
            # default bin size when generating the dataset
            original_scale = np.array([0.45, 0.54, 0.107])
            new_scale = self.table_full_size
            if mesh_name == 'cube' and not np.all(np.isclose(original_scale, new_scale)):
                goal[:2] = ((goal[:3] - self.table_offset)*new_scale/original_scale*0.5 + self.table_offset)[:2]

            # add variation to the goals when there is no wall interactions
            if self.goal_mode in {"any", "any_var_size"}:
                pos_range_lb = self.location_center - self.location_scale * 0.75
                pos_range_ub = self.location_center + self.location_scale * 0.75
                goal_xy = np.random.uniform(pos_range_lb, pos_range_ub)[:2]
                goal[:2] = goal_xy
                # goal[6] = np.random.uniform(-1, 1)
                # goal_pos[2] = obs['cube_pos'][2]

            goal = to_pose_mat(goal[:3], goal[3:])

        elif self.goal_mode == 'fixed':
            goal_pos = self.location_center
            goal_pos[2] = self.get_cube_size()[2] + self.table_offset[2]
            goal_ori = np.array([1., 0., 0., 0.])
            goal = to_pose_mat(goal_pos, goal_ori)
        
        elif self.goal_mode == 'fixed_off_center':
            goal = to_pose_mat(np.array([0.5, 0.1, obs['cube_pos'][2]]), np.array([1., 0., 0., 0.]))
        
        elif self.goal_mode == 'off_center':
            y = np.random.uniform(0.0, 0.2)
            goal = to_pose_mat(np.array([0.5, y, obs['cube_pos'][2]]), np.array([1., 0., 0., 0.]))
            
        elif self.goal_mode == 'translation':
            # Same pose as the object, with x, y randomized 
            # in the range of the bin (scaled by 0.6)
            pos_range_lb = self.location_center - self.location_scale * 0.6
            pos_range_ub = self.location_center + self.location_scale * 0.6
            goal_pos = np.random.uniform(pos_range_lb, pos_range_ub)
            goal_pos[2] = obs['cube_pos'][2]
            goal_ori = obs['cube_quat']
            goal = to_pose_mat(goal_pos, goal_ori)
        
        elif self.goal_mode == 'upright':
            # Sample a x,y location
            pos_range_lb = self.location_center - self.location_scale * 0.6
            pos_range_ub = self.location_center + self.location_scale * 0.6
            goal_pos = np.random.uniform(pos_range_lb, pos_range_ub)
            goal_pos[2] = self.get_cube_size()[2] + self.table_offset[2]
            
            # Sample a planar rotation
            # quat=cos(a/2),sin(a/2)⋅(x,y,z)
            z_angle = np.random.uniform(360)
            goal_ori = np.array([np.cos(z_angle/2), 0, 0, np.sin(z_angle/2)])
            goal = to_pose_mat(goal_pos, goal_ori)
            
        else:
            raise NotImplementedError
             
        return goal