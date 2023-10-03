import numpy as np
import torch
import hacman.utils.robosuite_transform_utils as transform_utils
import scipy.spatial.transform as spt
from transforms3d import quaternions, euler
# from torchgeometry.core.conversions import rotation_matrix_to_quaternion
from pytorch3d.transforms import matrix_to_quaternion

def decompose_pose_tensor(pose: torch.Tensor, cat=False):
    # Note: It handles torch tensors rather than ndarray
    pos = pose[:, :3, 3]
    quat = matrix_to_quaternion(pose[:, :3, :3])
    quat = quat[:, (1, 2, 3, 0)] # Move w to the last to match with robosuite

    # Verification only (uncomment this to debug)
    # test_p = pose[0].detach().cpu().numpy()
    # test_pos, test_quat = transform_utils.mat2pose(test_p) # This does not work on torch Tensors

    if cat:
        return torch.cat([pos, quat], dim=1)
    else:
        return pos, quat

def decompose_pose_mat(pose, cat=False, wxyz=True):
    pos, quat = transform_utils.mat2pose(pose)
    if wxyz:
        quat = transform_utils.convert_quat(quat, to="wxyz")
    if cat:
        return np.concatenate([pos, quat])
    else:
        return pos, quat


def to_pose_mat(pos, quat, input_wxyz=True):
    """
    Convert position and quaternion to a pose matrix.

    Args:
        pos (ndarray): Position vector of shape (3,) representing the translation.
        quat (ndarray): Quaternion vector of shape (4,) representing the orientation.
        input_wxyz (bool): Indicates whether the input quaternion follows the wxyz convention (True) or xyzw convention (False).

    Returns:
        ndarray: Pose matrix of shape (4, 4) representing the pose.
    """
    if not input_wxyz:
        quat = quat[[3, 0, 1, 2]]  # Convert wxyz to xyzw convention

    mat = quaternions.quat2mat(quat)
    pose = np.eye(4)
    pose[:3, :3] = mat
    pose[:3, 3] = pos
    return pose



def inv_pose_mat(pose):
    """
    Compute the inverse of a pose matrix.

    Args:
        pose (ndarray): Pose matrix of shape (4, 4).

    Returns:
        ndarray: Inverse pose matrix of shape (4, 4).
    """
    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def transform_point_cloud(current_pose_mat, next_pose_mat, pointcloud):
    """
    Transform a point cloud from the current pose to the next pose.

    This function takes the current and next pose matrices along with the point cloud and performs the transformation.
    The point cloud can be either batched or non-batched.

    Args:
        current_pose_mat (ndarray): Current pose matrix of shape (batch_size, 4, 4) or (4, 4).
        next_pose_mat (ndarray): Next pose matrix of shape (batch_size, 4, 4) or (4, 4).
        pointcloud (ndarray): Point cloud of shape (batch_size, num_points, 3) or (num_points, 3).

    Returns:
        ndarray: Transformed point cloud of shape (batch_size, num_points, 3) or (num_points, 3).
    """
    # Expand dimensions if input is non-batched
    if current_pose_mat.ndim == 2:
        current_pose_mat = np.expand_dims(current_pose_mat, axis=0)
        next_pose_mat = np.expand_dims(next_pose_mat, axis=0)
        pointcloud = np.expand_dims(pointcloud, axis=0)

    bs = current_pose_mat.shape[0]
    transformed_pcds = []
    for i in range(bs):
        transform = np.dot(next_pose_mat[i], inv_pose_mat(current_pose_mat[i]))

        transformed_pcd = np.concatenate([pointcloud[i], np.ones((len(pointcloud[i]), 1))], axis=1)
        transformed_pcd = np.dot(transform, transformed_pcd.T).T[:, :3]
        transformed_pcds.append(transformed_pcd)

    return np.array(transformed_pcds).squeeze()


def sample_idx(old_length, new_length):
    """
    Sample indices from the range [0, old_length) to obtain a new list of indices of length new_length.

    Args:
        old_length (int): The length of the original index range.
        new_length (int): The desired length of the sampled index list.

    Returns:
        ndarray: The list of sampled indices of length new_length.
    """
    if old_length > new_length:
        idx = np.random.choice(old_length, new_length, replace=False)
    elif old_length < new_length:
        idx = np.arange(new_length)
        idx[old_length:] = np.random.choice(old_length, new_length - old_length, replace=True)
    else:
        idx = np.arange(old_length)
        
    return idx
    