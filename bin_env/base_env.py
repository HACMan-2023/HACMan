from collections import OrderedDict
import numpy as np

from bin_env.assets import HousekeepSampler

from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from bin_env.bin_arena import BinArena
from bin_env.single_arm import SingleArm
from robosuite.utils import transform_utils

import copy

class BaseEnv(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the  initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        table_offset (3-tuple): x, y, and z location of the table top with respect to the robot base.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        initial_qpos=None,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.45, 0.54, 0.107),
        table_size_rand=(0., 0., 0.),
        table_friction=(0.3, 5e-3, 1e-4),
        table_solref=(0.02, 1),
        table_solimp=(0.9, 0.95, 0.001),
        table_offset=(0.5, 0, 0.065),
        table_xml="bin_arena.xml",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,      # {None, instance, class, element}
        adaptive=False,
        additional_obs_keys=None,
        close_gripper=False,
        remove_walls=False,
        object_dataset="cube",     # {"cube", "acronym", "shapenet", "housekeep"}
        object_types=None,
        object_name=None,
        object_split=None,
        object_scale_range=None,
        object_size_limit=[0.05, 0.05],
        object_init_pose="planar", #{"planar", "fixed", "any"}
        **kwargs
    ):
        if initial_qpos is None:
            self.initial_qpos = np.array([0, 0.15, 0, -2.44, 0, 2.62, -7.84e-01])
        else:
            self.initial_qpos = initial_qpos

        # settings for table top
        self.table_full_size = np.array(table_full_size)
        self.table_size_rand = np.array(table_size_rand)
        self.table_size_curr = np.array(table_full_size) # table size after randomization
        self.table_friction = np.array(table_friction)
        self.table_solref = np.array(table_solref)
        self.table_solimp = np.array(table_solimp)
        self.table_offset = np.array(table_offset)
        self.table_xml = table_xml

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer 
        # placement_initializer will be overwritten in _load_model
        self.placement_initializer = placement_initializer
        self.object_init_pose = object_init_pose
        self.object_scale = None
        
        # specify object
        self.object_dataset = object_dataset
        # self.object_types = object_types
        # self.object_name = object_name
        # self.object_split = object_split
        # self.object_eval_set = object_eval_set
        if object_dataset == 'housekeep' or object_dataset == 'housekeep_all':
            self.object_sampler = HousekeepSampler(
                object_types=object_types, object_name=object_name, 
                object_scale_range=object_scale_range,
                object_size_limit=object_size_limit,
                object_split=object_split, use_full_set=(object_dataset=='housekeep_all'))

        if not (kwargs is None or len(kwargs.keys()) == 0):
            print('Unused kwargs in DexEnv:', kwargs)

        # Additional Parameters for ADR
        self.object_to_wall_dist_min = 0
        self.object_to_wall_dist_max = 0
        self.object_pos_noise = 0
        self.object_ori_noise = 0
        self.table_offset_x_min = self.table_offset[0]
        self.table_offset_x_max = self.table_offset[0]
        self.table_offset_z_min = self.table_offset[2]
        self.table_offset_z_max = self.table_offset[2]

        self.object_size_x_min = 0.087
        self.object_size_x_max = 0.087
        self.object_size_x_val = 0.087
        self.object_size_y_min = 0.18
        self.object_size_y_max = 0.18
        self.object_size_y_val = 0.18
        self.object_size_z_min = 0.043
        self.object_size_z_max = 0.043
        self.object_size_z_val = 0.043
        self.object_density_min = 86.
        self.object_density_max = 86.
        self.object_density_val = 86.

        self.table_friction_min = self.table_friction[0]
        self.table_friction_max = self.table_friction[0]
        self.table_friction_val = self.table_friction[0]
        self.gripper_friction_min = 3
        self.gripper_friction_max = 3
        self.gripper_friction_val = 3
        self.close_gripper = close_gripper

        self.controller_max_translation_max = 0.03
        self.controller_max_translation_min = 0.03
        self.controller_max_rotation_max = 0.2
        self.controller_max_rotation_min = 0.2

        self.additional_obs_keys = additional_obs_keys if adaptive and additional_obs_keys else None
        self.remove_walls = remove_walls
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        if 'max_translation' in self.robots[0].controller_config:
            self.robots[0].controller_config['max_translation'] = np.random.uniform(self.controller_max_translation_min,
                                                                                    self.controller_max_translation_max)
        if 'max_rotation' in self.robots[0].controller_config:
            self.robots[0].controller_config['max_rotation'] = np.random.uniform(self.controller_max_rotation_min,
                                                                                 self.controller_max_rotation_max)

        # load model for table top workspace
        self.table_friction_val = np.random.uniform(self.table_friction_min, self.table_friction_max)
        self.table_friction[0] = self.table_friction_val
        self.table_offset[0] = np.random.uniform(low=self.table_offset_x_min,
                                                 high=self.table_offset_x_max)
        self.table_offset[2] = np.random.uniform(low=self.table_offset_z_min,
                                                 high=self.table_offset_z_max)
        if self.remove_walls:
            bin_full_size = np.array([0.8, 0.8, 0.107])
            mujoco_arena = BinArena(bin_pos=self.table_offset,
                                    bin_full_size=bin_full_size,
                                    bin_friction=self.table_friction,
                                    bin_solref=self.table_solref,
                                    bin_solimp=self.table_solimp,
                                    hidden_walls="FBLR",
                                    xml_filename=self.table_xml)
        else:
            self.table_size_curr = self.table_full_size + (np.random.rand(3)*2 - 1)*self.table_size_rand
            mujoco_arena = BinArena(bin_pos=self.table_offset,
                                    bin_full_size=self.table_size_curr,
                                    bin_friction=self.table_friction,
                                    bin_solref=self.table_solref,
                                    bin_solimp=self.table_solimp,
                                    hidden_walls="",
                                    xml_filename=self.table_xml)

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.object_size_x_val = np.random.uniform(self.object_size_x_min, self.object_size_x_max)
        self.object_size_y_val = np.random.uniform(self.object_size_y_min, self.object_size_y_max)
        self.object_size_z_val = np.random.uniform(self.object_size_z_min, self.object_size_z_max)
        self.object_density_val = np.random.randint(self.object_density_min, self.object_density_max+1)
        
        if self.object_dataset == "cube":
            self.cube = BoxObject(
                name="cube",
                size_min=[self.object_size_x_val/2, self.object_size_y_val/2, self.object_size_z_val/2],
                size_max=[self.object_size_x_val/2, self.object_size_y_val/2, self.object_size_z_val/2],
                rgba=[1, 0, 0, 1],
                material=redwood,
                density=self.object_density_val,
            )
            
            self.cube_target = BoxObject(
                name="cube_target",
                size_min=[self.object_size_x_val/2, self.object_size_y_val/2, self.object_size_z_val/2],
                size_max=[self.object_size_x_val/2, self.object_size_y_val/2, self.object_size_z_val/2],
                rgba=[0, 1, 0, 0.1],
                joints=None,
                obj_type="visual",
            )
        
        elif self.object_dataset == "housekeep" or self.object_dataset == "housekeep_all":
            obj, target_obj = self.object_sampler.generate_housekeep_object()
            self.cube = obj
            self.cube_target = target_obj
        
        else:
            raise ValueError(f"Unknown object type: {self.object_dataset}")

        # Create placement initializer
        if self.object_init_pose == "planar":
            max_dimension = max(self.object_size_x_val, self.object_size_y_val, self.object_size_z_val)
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-self.table_size_curr[0]/2+max_dimension/2, self.table_size_curr[0]/2-max_dimension/2],
                y_range=[-self.table_size_curr[1]/2+max_dimension/2, self.table_size_curr[1]/2-max_dimension/2],
                rotation=None,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,  # Problematic
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.05,
            )
        elif self.object_init_pose == "fixed":
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[0, 0],
                y_range=[0, 0],
                rotation=0,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,  # Problematic
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.05,
            )
        elif self.object_init_pose == "any-ori":
            max_dimension = max(self.object_size_x_val, self.object_size_y_val, self.object_size_z_val)
            self.placement_initializer = CustomizedSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[0, 0],
                y_range=[0, 0],
                rotation=None,
                rotation_axis='all',
                ensure_object_boundary_in_range=False,  # Problematic
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.05,
            )
        elif self.object_init_pose == "any":
            max_dimension = max(self.object_size_x_val, self.object_size_y_val, self.object_size_z_val)
            self.placement_initializer = CustomizedSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-self.table_size_curr[0]/2+max_dimension/2, self.table_size_curr[0]/2-max_dimension/2],
                y_range=[-self.table_size_curr[1]/2+max_dimension/2, self.table_size_curr[1]/2-max_dimension/2],
                rotation=None,
                rotation_axis='all',
                ensure_object_boundary_in_range=False,  # Problematic
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.05,
            )
        else:
            raise NotImplementedError

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[self.cube, self.cube_target],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return np.array(self.sim.data.body_xquat[self.cube_body_id])
            
            @sensor(modality=modality)
            def cube_size(obs_cache):
                return np.array([self.object_size_x_val, self.object_size_y_val, self.object_size_z_val])

            sensors = [cube_pos, cube_quat, cube_size]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables
        
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _load_robots(self):
        gripper_friction = np.array([3, 0.015, 0.0003])
        self.gripper_friction_val = np.random.uniform(self.gripper_friction_min, self.gripper_friction_max)
        gripper_friction[0] = self.gripper_friction_val
        if self.robot_names[0] == "Virtual":
            self.robot_configs[0].update({'mount_type':None})
        self.robots[0] = SingleArm(robot_type=self.robot_names[0], idn=0, control_gripper=False,
                                   initial_qpos=self.initial_qpos, gripper_friction=gripper_friction,
                                   **self.robot_configs[0], close_gripper=self.close_gripper)
        self.robots[0].load_model()


class CustomizedSampler(UniformRandomSampler):
    def _sample_quat(self):
        if self.rotation_axis == 'all':
            return transform_utils.random_quat()
        else:
            super()._sample_quat()