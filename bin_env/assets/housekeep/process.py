import open3d as o3d
import os
import h5py
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns

root_dir = os.path.dirname(os.path.abspath(__file__))
dirname = 'bin_env/assets/housekeep/models'
processed_dirname = 'bin_env/assets/housekeep/models_processed/meshes'
info_dirname = 'bin_env/assets/housekeep/object_info'
v_hacd_path = '../v-hacd/app/build/TestVHACD'
manifold_path = '../Manifold/build/manifold'

def process_objs():
    for obj_file in os.listdir(dirname):
        if obj_file[-4:] != '.obj':
            continue
        
        print(obj_file)
        obj_path = os.path.join(dirname, obj_file)
        
        # v-hacd
        v_hacd_dir = os.path.dirname(v_hacd_path)
        os.system(f'{v_hacd_path} {obj_path}')
        
        # manifold
        os.system(f'{manifold_path} decomp.obj decomp.obj 20000')
        
        mesh = o3d.io.read_triangle_mesh("decomp.obj", True)
        # mesh = o3d.io.read_triangle_mesh(obj_path, True)
        mesh.compute_vertex_normals()
        # pcd = o3d.geometry.sample_points_uniformly(mesh, number_of_points=10000)
        # points = np.asarray(pcd.points)
        # pcd_scaled = o3d.geometry.PointCloud()
        # pcd_scaled.points = o3d.utility.Vector3dVector(points)
        vertices = np.asarray(mesh.vertices)
        print(vertices.shape)
        print("Water tight? ", mesh.is_watertight())

        max_points = np.max(vertices, axis=0)
        min_points = np.min(vertices, axis=0)
        print(max_points)
        print(min_points)
        size = (max_points-min_points)/2
        origin = (max_points+min_points)/2
        
        stl_file = obj_file[:-4] + '.stl'
        stl_path = os.path.join(processed_dirname, stl_file)
        # o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh(stl_path, mesh)
        
        for f in ["decomp.mtl", "decomp.obj", "decomp.stl"]:
            if os.path.exists(f):
                os.remove(f)
        
        filename = stl_file[:-4] +".h5"
        filepath = os.path.join(info_dirname, filename)
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("origin", data=origin)
            f.create_dataset("size", data=size)

def generate_metadata():
    metadata = {
        'name': [],
        'type': [],
    }
    
    for h5_file in os.listdir(info_dirname):
        if h5_file[-3:] != '.h5':
            continue
        
        obj_name = h5_file[:-3]
        obj_type = obj_name.split('_')[0]
        metadata['name'].append(obj_name)
        metadata['type'].append(obj_type)
        
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(root_dir, 'metainfo.csv'))
    return df

def generate_dataset_split(df: pd.DataFrame):
    # Split into train, unseen instance, unseen class

    # Unseen instance: 2 objects per type from the top 5 classes:
    # Cup, mug, bowl, bottle, plate
    unseen_instance_classes = ['cup', 'mug', 'bottle', 'pillbottle']
    unseen_instances = []
    for obj_type in unseen_instance_classes:
        obj_names = df[df['type'] == obj_type]['name'].values
        unseen_instances.extend(np.random.choice(obj_names, 1, replace=False))

    # Unseen class: 15 objects from the rest of the classes
    # rest_classes = np.setdiff1d(df['type'].unique(), unseen_instance_classes)
    # unseen_class_objs = []
    # while len(unseen_class_objs) != 15:
    #     unseen_classes = np.random.choice(rest_classes, 7, replace=False)
    #     unseen_class_objs = df[df['type'].isin(unseen_classes)]['name'].values
    unseen_class = "cup"
    unseen_class_objs = df[df['type'] == unseen_class]['name'].values
    
    # Train: the rest
    test_objs = np.concatenate([unseen_instances, unseen_class_objs])
    train_objs = np.setdiff1d(df['name'].values, test_objs)

    data = {
        "train": train_objs.tolist(),
        "unseen_instance": unseen_instances,
        "unseen_class": unseen_class_objs.tolist()
    }
    print(f"Dataset split: {len(train_objs)} train, {len(unseen_instances)} unseen instance, {len(unseen_class_objs)} unseen class")

    with open(os.path.join(root_dir, 'dataset_split.json'), 'w') as f:
        json.dump(data, f, indent=4)

    return

def plot_type_histogram(df: pd.DataFrame):
    """
    Plot the histogram of number of objects per type
    """
    plt.figure(figsize=(10, 5))
    sns.countplot(x='type', data=df)
    plt.title('Number of objects per type')
    # plt.savefig(os.path.join(root_dir, 'type_histogram.png'))
    plt.show()
    return
    

def generate_unseen_instance_split(df: pd.DataFrame):
    # per type
    all_types = df['type'].unique()
    test_set = []
    train_set = []
    
    for obj_type in all_types:
        obj_names = df[df['type'] == obj_type]['name'].values
        test_objs = np.random.choice(obj_names, 1, replace=False)
        train_objs = np.setdiff1d(obj_names, test_objs)
        
        test_set.extend(test_objs)
        train_set.extend(train_objs)
    
    data = {
        "train": train_set,
        "test": test_set
    }
    print(f"Unseen instance split: {len(test_set)} test, {len(train_set)} train")
    with open(os.path.join(root_dir, 'unseen_instance_split.json'), 'w') as f:
        json.dump(data, f, indent=2)
    
    return

def generate_unseen_type_split(df: pd.DataFrame, types=['cup']):
    all_types = df['type'].unique()
    if types is not None:
        test_types = types
    else:
        num_test_type = int(np.ceil(len(all_types) * 0.2)) 
        test_types = np.random.choice(all_types, num_test_type, replace=False)
    
    test_set = df[df['type'].isin(test_types)]['name'].values
    train_set = np.setdiff1d(df['name'].values, test_set)
    
    data = {
        "train": train_set.tolist(),
        "test": test_set.tolist()
    }
    print(f"Unseen type split: {len(test_set)} test, {len(train_set)} train")
    with open(os.path.join(root_dir, 'unseen_type_split.json'), 'w') as f:
        json.dump(data, f, indent=2)
    return
        

def generated_mixed_split(df: pd.DataFrame):
    num_obj = len(df)
    num_test_obj = int(np.ceil(num_obj * 0.2))
    
    test_set = np.random.choice(df['name'].values, num_test_obj, replace=False)
    train_set = np.setdiff1d(df['name'].values, test_set)
    
    data = {
        "train": train_set.tolist(),
        "test": test_set.tolist()
    }
    print(f"Mixed split: {len(test_set)} test, {len(train_set)} train")
    with open(os.path.join(root_dir, 'mixed_split.json'), 'w') as f:
        json.dump(data, f, indent=2)
    return

def test_read():
    df = pd.read_csv(os.path.join(root_dir, 'metainfo.csv'))
    print(df['name'].values)

# pca = PCA()
# pca.fit(points)
# eign_vec = pca.components_
# pcd_extra = o3d.geometry.PointCloud()
# # origin = np.mean(points, axis=0).reshape(1,-1)
# # print(origin)
# # print(data['object/com'][()])
# extra_points = []
# extra_points.append(np.array([[max_points[0], 0, 0]]))
# extra_points.append(np.array([[min_points[0], 0, 0]]))
# # for i in range(10):
# #     # extra_points.append(origin+eign_vec*i*0.01)
#
# print(eign_vec)
#
# extra_points = np.concatenate(extra_points)
# pcd_extra.points = o3d.utility.Vector3dVector(extra_points)
#
# o3d.visualization.draw_geometries([pcd_scaled, pcd_extra])

if __name__ == "__main__":
    # process_objs()
    df = generate_metadata()
    plot_type_histogram(df)
    generate_dataset_split(df)
    # generate_unseen_instance_split(df)
    # generate_unseen_type_split(df)
    # generated_mixed_split(df)
    test_read()
