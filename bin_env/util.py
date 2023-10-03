from bin_env.utils.rotations import *
import numpy as np
import robosuite.utils.transform_utils as T

def angle_diff(quat_a, quat_b):
    # Subtract quaternions and extract angle between them.
    quat_diff = quat_mul(quat_a, quat_conjugate(quat_b))
    a_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
    return min(a_diff, np.pi*2 - a_diff)


def convert_to_batch(arr):
    if len(arr.shape) == 1:
        arr = arr[np.newaxis]
    return arr


def quat_rot_vec_arr(q, v0):
    q_v0 = np.array([np.zeros_like(v0[...,0]), v0[...,0], v0[...,1], v0[...,2]]).transpose()
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[..., 1:]
    return v


def get_global_pose(frame_pos, frame_quat, pos_in_frame, quat_in_frame):
    # Convert the grasp pose in the object coordinate to the global coordinate
    quat = quat_mul(frame_quat, quat_in_frame)
    pos = quat_rot_vec_arr(frame_quat, pos_in_frame) + frame_pos
    return pos, quat


def get_local_pose(frame_pos, frame_quat, global_pos, global_quat):
    # Convert the grasp pose in the global coordinate to the object coordinate
    local_quat = quat_mul(quat_conjugate(frame_quat), global_quat)
    local_pos = quat_rot_vec_arr(quat_conjugate(frame_quat), global_pos - frame_pos)
    return local_pos, local_quat


def clean_xzplane_pose(pos, quat, gripper_correction=False):
    if gripper_correction:
        rotate_back = euler2quat(np.array([0, -np.pi / 2, 0]))  # rotate 180 degree
        quat = quat_mul(quat, rotate_back)
    euler = quat2euler(quat)
    return np.array([pos[0], pos[2], euler[1]])


def clean_4d_pose(pos, quat, gripper_correction=False):
    if gripper_correction:
        rotate_back = euler2quat(np.array([0, -np.pi / 2, 0]))  # rotate 180 degree
        quat = quat_mul(quat, rotate_back)
    euler = quat2euler(quat)
    return np.array([pos[0], pos[1], pos[2], euler[1]])


def clean_6d_pose(pos, quat, gripper_correction=False):
    return np.concatenate([pos, quat])


def sample_unit_vector():
    v = np.random.randn(3)
    v /= np.linalg.norm(v)
    return v


def add_pose_noise(pos, quat, pos_noise=0, ori_noise=0):
    noisy_pos = pos + sample_unit_vector() * pos_noise

    axis = sample_unit_vector()
    angle = np.random.uniform(0, ori_noise/180*np.pi)
    additional_quat = np.array([np.cos(angle/2), np.sin(angle)*axis[0], np.sin(angle)*axis[1], np.sin(angle)*axis[2]]) #wxyz
    noisy_quat = quat_mul(additional_quat, quat)
    return noisy_pos, noisy_quat

def get_error(R_list, t_list, data):
    "given the predicted translation and rotation, compute the error"
    rotation_loss_list = []
    translation_loss_list = []
    succ_list = []
    for k, rotation in enumerate(R_list):
        current_obj_mat = data['current_obj_pose'].reshape(-1,4,4)[k][:3,:3]
        current_obj_position = data['current_obj_pose'].reshape(-1,4,4)[k][:3,3]
        next_obj_mat = data['next_obj_pose'].reshape(-1,4,4)[k][:3,:3]
        next_obj_position = data['next_obj_pose'].reshape(-1,4,4)[k][:3,3]
        pred_next_obj_mat = R_list[k] @ current_obj_mat
        rotation_error = angle_diff(T.mat2quat(pred_next_obj_mat.cpu().numpy()), T.mat2quat(next_obj_mat.cpu().numpy()))/np.pi*180
        translation_error = np.linalg.norm((R_list[k]@current_obj_position + t_list[k] - next_obj_position).cpu().numpy())
        succ = np.float64(translation_error<0.03 and rotation_error<10)
        rotation_loss_list.append(rotation_error)
        translation_loss_list.append(translation_error)
        succ_list.append(succ)
    rotation_loss = np.array(rotation_loss_list).mean()
    translation_loss = np.array(translation_loss_list).mean()
    success_rate = np.array(succ_list).mean()
    return rotation_loss, translation_loss, success_rate

def get_gt_transformation_scale(R_list, t_list, data):
    current_obj_position = data['current_obj_pose'].reshape(-1,4,4)
    batch_size = current_obj_position.shape[0]
    gt_rotation_scale = []
    gt_translation_scale = []
    for k in range(batch_size):
        current_obj_mat = data['current_obj_pose'].reshape(-1,4,4)[k][:3,:3]
        current_obj_position = data['current_obj_pose'].reshape(-1,4,4)[k][:3,3]
        next_obj_mat = data['next_obj_pose'].reshape(-1,4,4)[k][:3,:3]
        next_obj_position = data['next_obj_pose'].reshape(-1,4,4)[k][:3,3]
        rotation_diff = angle_diff(T.mat2quat(next_obj_mat.cpu().numpy()), T.mat2quat(current_obj_mat.cpu().numpy()))/np.pi*180
        translation_diff = np.linalg.norm((next_obj_position - current_obj_position).cpu().numpy())
        gt_rotation_scale.append(rotation_diff)
        gt_translation_scale.append(translation_diff)
    rotation_loss_list = []
    translation_loss_list = []
    for k, rotation in enumerate(R_list):
        current_obj_mat = data['current_obj_pose'].reshape(-1,4,4)[k][:3,:3]
        current_obj_position = data['current_obj_pose'].reshape(-1,4,4)[k][:3,3]
        next_obj_mat = data['next_obj_pose'].reshape(-1,4,4)[k][:3,:3]
        next_obj_position = data['next_obj_pose'].reshape(-1,4,4)[k][:3,3]
        pred_next_obj_mat = R_list[k] @ current_obj_mat
        rotation_error = angle_diff(T.mat2quat(pred_next_obj_mat.cpu().numpy()), T.mat2quat(next_obj_mat.cpu().numpy()))/np.pi*180
        translation_error = np.linalg.norm((R_list[k]@current_obj_position + t_list[k] - next_obj_position).cpu().numpy())
        rotation_loss_list.append(rotation_error)
        translation_loss_list.append(translation_error)
    return gt_rotation_scale, rotation_loss_list, gt_translation_scale, translation_loss_list
