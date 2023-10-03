import numpy as np
import torch
from robosuite.utils import transform_utils
import scipy.spatial.transform as spt


def calculate_non_z_angle_diff(a_, b_, input_wxyz):
    a = np.copy(a_)
    b = np.copy(b_)

    # Convert to xyzw
    if input_wxyz:
        a = a[[1, 2, 3, 0]]
        b = b[[1, 2, 3, 0]]
    
    # Calculate the difference
    a = spt.Rotation.from_quat(a)
    b = spt.Rotation.from_quat(b)

    z = np.array([0, 0, 1])
    a_z = a.apply(z)
    b_z = b.apply(z)

    rad_diff = np.arccos(a_z * b_z / (np.linalg.norm(a_z) * np.linalg.norm(b_z)))
    diff = rad_diff * 180 / np.pi
    
    return diff


def calculate_axis_diff(a_, b_, input_wxyz, axis='z', output_wxyz=True):
    a = np.copy(a_)
    b = np.copy(b_)

    # Convert to xyzw
    if input_wxyz:
        a = a[[1, 2, 3, 0]]
        b = b[[1, 2, 3, 0]]
    
    # Calculate the difference
    a = spt.Rotation.from_quat(a)
    b = spt.Rotation.from_quat(b)
    diff = a.inv() * b

    if axis == 'z':
        z = np.array([0, 0, 1])

    new_z = diff.apply(z)

    # Calculate the angle difference
    rad_diff = np.arccos(new_z * z / (np.linalg.norm(new_z) * np.linalg.norm(z)))
    diff = rad_diff * 180 / np.pi

    return diff

def decompose_pose_mat(pose, cat=False, wxyz=True):
    pos, quat = transform_utils.mat2pose(pose)
    if wxyz:
        quat = transform_utils.convert_quat(quat, to="wxyz")
    if cat:
        return np.concatenate([pos, quat])
    else:
        return pos, quat

def to_pose_mat(pos, quat):
    # Input quat follows wxyz convention (MuJoCo)
    mat = transform_utils.quat2mat(transform_utils.convert_quat(quat))
    pose = transform_utils.make_pose(pos, mat)
    return pose


def transform_point_cloud(current_object_pose, next_object_pose, pointcloud):
    if len(current_object_pose.shape) <= 2:
        current_object_pose = np.expand_dims(current_object_pose, axis=0)
        next_object_pose = np.expand_dims(next_object_pose, axis=0)
        pointcloud = np.expand_dims(pointcloud, axis=0)
    
    bs = current_object_pose.shape[0]
    transformed_pcds = []
    for i in range(bs):
        transform = next_object_pose[i].dot(transform_utils.pose_inv(current_object_pose[i]))
    
        transformed_pcd = np.ones((len(pointcloud[i]), 4))
        transformed_pcd[:, :3] = pointcloud[i]
        transformed_pcd = transform.dot(transformed_pcd.transpose()).transpose()[:, :3]
        transformed_pcds.append(transformed_pcd)
    
    if len(transformed_pcds) == 1:
        transformed_pcd = transformed_pcds[0]
        return transformed_pcd
    
    else:
        transformed_pcds = np.stack(transformed_pcds, axis=0)
        return transformed_pcds