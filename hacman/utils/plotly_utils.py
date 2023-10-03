import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def format_color(color):
    if type(color) == str:
        color_str = color
    elif type(color) == np.ndarray and len(color) == 3:
        color = color.astype(np.int32)
        color_str = f'rgb({color[0]},{color[1]},{color[2]})'
    else:
        color = color.astype(np.int32)
        color_str = [f'rgb({color[k, 0]},{color[k, 1]},{color[k, 2]})' for k in range(color.shape[0])]
    return color_str

def plot_pcd(name, points, color, size=3):
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", 
                        marker = dict(color=format_color(color), size=size), name=name)

def plot_pcd_with_score(name, points, action_score, size=3):
    assert type(action_score) == np.ndarray
    action_score = action_score.reshape(-1)
    action_score = (action_score - action_score.min())/(action_score.max()-action_score.min()+1e-7)
    object_color_id = (action_score*255).astype(np.int32)
    object_color = []
    for cid in object_color_id:
        try:
            object_color.append(plt.get_cmap('plasma').colors[cid])
        except:
            print(f'cid={cid} gives an error. Will use 0 instead.')
            object_color.append(plt.get_cmap('plasma').colors[0])
    object_color = np.array(object_color)*255
    return plot_pcd(name, points, object_color, size)

def plot_action(name, start, direction, color='red', size=3):
    direction = direction*0.02*3  # action scale=0.02. steps=10
    if start is None: x, y, z = 0, 0, 0
    else: x, y, z = start[0], start[1], start[2]
    u, v, w = direction[0], direction[1], direction[2]
    return go.Scatter3d(x=[x, x + u], y=[y, y + v], z=[z, z + w], mode='lines',
                        line=dict(color=format_color(color), width=10), name=name)