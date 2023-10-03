import numpy as np
from bin_env.utils.rotations import *
import cv2


# -------------------- Generic ----------------------------
def get_intrinsics(fovy, img_width, img_height):
    # fovy = self.sim.model.cam_fovy[cam_no]
    aspect = float(img_width) / img_height
    fovx = 2 * np.arctan(np.tan(np.deg2rad(fovy) * 0.5) * aspect)
    fovx = np.rad2deg(fovx)
    cx = img_width / 2.
    cy = img_height / 2.
    fx = cx / np.tan(np.deg2rad(fovx / 2.))
    fy = cy / np.tan(np.deg2rad(fovy / 2.))
    K = np.zeros((3,3), dtype=np.float64)
    K[2][2] = 1
    K[0][0] = fx
    K[1][1] = fy
    K[0][2] = cx
    K[1][2] = cy
    return K


def depth2xyz(depth, cam_K):
    h, w = depth.shape
    ymap, xmap = np.meshgrid(np.arange(w), np.arange(h))

    x = ymap
    y = xmap
    z = depth

    x = (x - cam_K[0,2]) * z / cam_K[0,0]
    y = (y - cam_K[1,2]) * z / cam_K[1,1]

    xyz = np.stack([x, y, z], axis=2)
    return xyz


def visualize_point_cloud_from_nparray(d, c=None, vis_coordinate=False):
    if c is not None:
        if len(c.shape) == 3:
            c = c.reshape(-1, 3)
        if c.max() > 1:
            c = c.astype(np.float64)/256

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d)
    if c is not None:
        pcd.colors = o3d.utility.Vector3dVector(c)

    if vis_coordinate:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([mesh, pcd])
    else:
        o3d.visualization.draw_geometries([pcd])


# -------------------- MuJoCo Specific ----------------------------
def get_transformation_matrix(pos, quat):
    arr = np.identity(4)
    arr[:3, :3] = quat2mat(quat)
    arr[:3, 3] = pos
    return arr


def get_transformation(env, camera_name=None):
    if camera_name is None:
        camera_name = env.camera_names[0]
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = env.sim.model.cam_pos[cam_id]
    cam_quat = env.sim.model.cam_quat[cam_id]
    cam_quat = quat_mul(cam_quat, euler2quat([np.pi, 0, 0]))
    return get_transformation_matrix(cam_pos, cam_quat)


def convert_depth(env, depth):
    # Convert depth into meter
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    depth_m = depth * 2 - 1
    depth_m = (2 * near * far) / (far + near - depth_m * (far - near))
    # Check this as well: https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L734
    return depth_m


def get_object_point_cloud(env, depth, img):
    depth = convert_depth(env, depth)
    full_pc = get_point_cloud(env, depth)
    obj_mask = get_obj_mask(img)
    pc = full_pc[obj_mask.reshape(-1),:]
    return pc


def get_point_cloud(env, depth, camera_name=None):
    # make sure to convert the raw depth image from MuJoCo using convert_depth
    # output is flattened
    if camera_name is None:
        camera_name = env.camera_names[0]
    fovy = env.sim.model.cam_fovy[env.sim.model.camera_name2id(camera_name)]
    K = get_intrinsics(fovy, depth.shape[0], depth.shape[1])
    pc = depth2xyz(depth, K)
    pc = pc.reshape(-1, 3)

    transform = get_transformation(env, camera_name=camera_name)
    new_pc = np.ones((pc.shape[0], 4))
    new_pc[:, :3] = pc
    new_pc = (transform @ new_pc.transpose()).transpose()
    return new_pc[:, :-1]


def get_obj_mask(img):
    # Given an [n,n,3] image, output a [n,n] mask of the red object in the scene
    # Assume the object is red
    img = img[:, :, -1::-1]  # RGB to BGR
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # # set my output img to zero everywhere except my mask
    # output_img = img.copy()
    # output_img[np.where(mask == 0)] = 0
    # cv2.imwrite('color_img.jpg', output_img)
    return mask != 0


def add_additive_noise_to_xyz(
    xyz_img,
    gp_rescale_factor_range=[12, 20],
    gaussian_scale_range=[0.0, 0.003],
    valid_mask=None,
):
    """Add (approximate) Gaussian Process noise to ordered point cloud
    @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    gp_rescale_factor = np.random.randint(
        gp_rescale_factor_range[0], gp_rescale_factor_range[1]
    )
    gp_scale = np.random.uniform(gaussian_scale_range[0], gaussian_scale_range[1])

    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(
        loc=0.0, scale=gp_scale, size=(small_H, small_W, C)
    )
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    if valid_mask is not None:
        # use this to add to the image
        xyz_img[valid_mask, :] += additive_noise[valid_mask, :]
    else:
        xyz_img += additive_noise

    return xyz_img


def dropout_random_ellipses(
    depth_img, dropout_mean, gamma_shape=10000, gamma_scale=0.0001
):
    """Randomly drop a few ellipses in the image for robustness.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [s] set of depth z values
    """
    depth_img = depth_img.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(dropout_mean)

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(
        np.where(depth_img > 0)
    ).T  # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(
        nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
    )
    dropout_centers = nonzero_pixel_indices[
        dropout_centers_indices, :
    ]  # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        # mask is always 2d even if input is not
        mask = np.zeros(depth_img.shape[:2])
        mask = cv2.ellipse(
            mask,
            tuple(center[::-1]),
            (x_radius, y_radius),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
        )
        depth_img[mask == 1] = 0

    return depth_img
