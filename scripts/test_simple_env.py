# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import open3d as o3d

def test_simple_cube_sim():
    # Import the SimpleCubeSim class and create an instance
    from hacman.envs.simple_env import SimpleCubeSim
    sim = SimpleCubeSim()
    sim.reset(np.zeros(2))

    # Get the point cloud and segmentation before the poke
    point_cloud_before, segmentation_before = sim.get_point_cloud()

    # Perform a poke on the cube
    location = [0.1, 0.1, 0.1]  # Example poke location
    motion = [-0.2, -0.4, 0.0]   # Example poke motion
    sim.poke(location, motion)

    # Get the point cloud and segmentation after the poke
    point_cloud_after, segmentation_after = sim.get_point_cloud()

    # Visualize the point clouds with segmentation color
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    cmap = ListedColormap(['#1f77b4', '#f2f2f2'])

    # Before poke
    ax1.scatter(point_cloud_before[:, 0], point_cloud_before[:, 1], c=segmentation_before, cmap=cmap)
    ax1.scatter(location[0], location[1], c='C2', marker='o', label='Contact Location')
    ax1.scatter(location[0]+motion[0], location[1]+motion[1], c='C1', marker='.', label='__nolegend__')
    ax1.quiver(location[0], location[1], motion[0], motion[1], color='C1', label='Motion')
    ax1.set_title('Point Cloud Before Poke')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    # After poke
    ax2.scatter(point_cloud_after[:, 0], point_cloud_after[:, 1], c=segmentation_after, cmap=cmap)
    ax2.set_title('Point Cloud After Poke')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    plt.show()


def test_simple_env_2d(trials=10):
    # Import the SimpleEnv class
    from hacman.envs.sim_envs.simple_env import SimpleEnv

    # Create a SimpleEnv object
    env = SimpleEnv()

    for i in range(trials):
        # Reset the environment
        obs = env.reset()

        # Store initial observation
        initial_object_pcd_points = obs['object_pcd_points']
        initial_background_pcd_points = obs['background_pcd_points']

        # Perform a step in the environment
        action = np.random.random(3) * 2 - 1
        next_obs, reward, done, info = env.step(action)

        # Store updated observation
        updated_object_pcd_points = next_obs['object_pcd_points']
        updated_background_pcd_points = next_obs['background_pcd_points']

        # Create matplotlib figure and axes
        plt.subplots(figsize=(7, 7))
        plt.xlabel("X")
        plt.ylabel("Y")

        # Plot points
        plt.scatter(initial_background_pcd_points[:, 0], initial_background_pcd_points[:, 1], color='lightgrey', label="Previous Background Points")
        plt.scatter(updated_background_pcd_points[:, 0], updated_background_pcd_points[:, 1], color='grey', label="Current Background Points")
        plt.scatter(initial_object_pcd_points[:, 0], initial_object_pcd_points[:, 1], color='orange', label="Previous Object Points")
        plt.scatter(updated_object_pcd_points[:, 0], updated_object_pcd_points[:, 1], color='blue', label="Current Object Points")
        

        # Plot the action line
        action_location = np.round(info["action_location"], 2)
        action_param = np.round(info["action_param"], 2)
        action_endpoint = action_location + action_param
        plt.plot([action_location[0], action_endpoint[0]], [action_location[1], action_endpoint[1]], color='red', label="Action")

        # Show the plot
        plt.title(f"Point Cloud Visualization \n Location: {action_location} \n Param: {action_param} \n Reward: {reward}")
        plt.legend()
        plt.show()

def test_simple_env_3d(trials=10):
    # Create a SimpleEnv object
    from hacman.envs.sim_envs.simple_env import SimpleEnv
    env = SimpleEnv()

    for _ in range(trials):
        # Reset the environment
        obs = env.reset()

        # Store initial observation
        initial_object_pcd_points = obs['object_pcd_points']
        initial_background_pcd_points = obs['background_pcd_points']

        # Create Open3D point cloud objects for initial observation
        initial_object_pcd = o3d.geometry.PointCloud()
        initial_object_pcd.points = o3d.utility.Vector3dVector(initial_object_pcd_points)
        initial_object_pcd.paint_uniform_color([1.0, 0.65, 0.0])  # Orange color for previous object points

        initial_background_pcd = o3d.geometry.PointCloud()
        initial_background_pcd.points = o3d.utility.Vector3dVector(initial_background_pcd_points)
        initial_background_pcd.paint_uniform_color([0.9, 0.9, 0.9])  # Light grey color for previous background points

        # Visualize the initial point clouds using Open3D
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(initial_background_pcd)
        visualizer.add_geometry(initial_object_pcd)

        # Perform a step in the environment
        action = np.random.random(3) * 2 - 1
        next_obs, reward, done, info = env.step(action)
        print("Reward: ", reward)

        # Store updated observation
        updated_object_pcd_points = next_obs['object_pcd_points']
        updated_background_pcd_points = next_obs['background_pcd_points']

        # Create Open3D point cloud objects for updated observation
        updated_object_pcd = o3d.geometry.PointCloud()
        updated_object_pcd.points = o3d.utility.Vector3dVector(updated_object_pcd_points)
        updated_object_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue color for current object points

        updated_background_pcd = o3d.geometry.PointCloud()
        updated_background_pcd.points = o3d.utility.Vector3dVector(updated_background_pcd_points)
        updated_background_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Grey color for current background points

        # Create an Open3D line segment to represent the action
        action_location = info["action_location"]
        action_param = info["action_param"]
        action_endpoint = action_location + action_param

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([action_location, action_endpoint])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # Red color for the action line

        # Visualize the updated point clouds and action line using Open3D
        visualizer.add_geometry(updated_background_pcd)
        visualizer.add_geometry(updated_object_pcd)
        visualizer.add_geometry(line)
        visualizer.run()
        visualizer.destroy_window()

if __name__ == '__main__':
    # test_simple_cube_sim()
    test_simple_env_2d()
    # test_simple_env_3d()