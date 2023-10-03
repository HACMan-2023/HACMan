import open3d as o3d
import os
import h5py
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")

root_dir = os.path.dirname(os.path.abspath(__file__))
dirname = 'bin_env/assets/housekeep_all/models'
processed_dirname = 'bin_env/assets/housekeep_all/models_processed/meshes'
info_dirname = 'bin_env/assets/housekeep_all/object_info'
v_hacd_path = '../v-hacd/app/build/TestVHACD'
manifold_path = '../Manifold/build/manifold'

def process_objs():
    for obj_dir in tqdm(os.listdir(dirname)):
        if obj_dir[-2:] != '_M':
            continue
        
        obj_type = obj_dir.split('_')[0]
        if obj_type in ['fork', 'knife', 'scissors', 'spoon', 'screwdriver']:
            continue
        # if obj_file[-4:] != '.obj':
        #     continue
        
        print(obj_dir)
        obj_path = os.path.join(dirname, obj_dir, 'model.obj')
        
        # v-hacd
        # v_hacd_dir = os.path.dirname(v_hacd_path)
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
        
        stl_file = obj_dir + '.stl'
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
    unstable_objs = [
        "handheld_game_console_RedBlack_Nintendo_3DSXL_M",
        "hat_DPC_Handmade_Hat_Brown_M",
        "can_opener_OXO_Soft_Works_Can_Opener_SnapLock_M",
        "tray_Design_Ideas_Drawer_Store_Organizer_M",
        "lego_block_073-c_lego_duplo_M",
        "keyboard_Kanex_MultiSync_Wireless_Keyboard_M",
        "clock_frl_apartment_clock_M",
        "hard_drive_Seagate_1TB_Backup_Plus_portable_drive_Blue_M",
        "lego_block_073-a_lego_duplo_M",
        "mouse_Razer_Naga_MMO_Gaming_Mouse_M",
        "box_Little_Debbie_Donut_Sticks_6_cake_donuts_10_oz_total_M",
        "keyboard_Razer_Blackwidow_Tournament_Edition_Keyboard_M",
        "magnifying_glass_Magnifying_Glassassrt_M",
        "pitcher_019_pitcher_base_M",
        "book_frl_apartment_book_02_M",
        "bowl_kitchen_set_kitchen_set_soup_bowl_M",
        "hair_straightener_Remington_TStudio_Silk_Ceramic_Hair_Straightener_2_Inch_Floating_Plates_M",
        "tray_Ecoforms_Saucer_SQ3_Turquoise_M",
        ]
    unstable_types = [
        "plate"
    ]
    
    metadata = {
        'name': [],
        'type': [],
    }
    
    for h5_file in os.listdir(info_dirname):
        if h5_file[-3:] != '.h5':
            continue
        
        obj_name = h5_file[:-3]
        obj_type = obj_name.split('_')[0]
        
        if obj_name in unstable_objs or obj_type in unstable_types:
            continue

        metadata['name'].append(obj_name)
        metadata['type'].append(obj_type)
        
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(root_dir, 'metainfo.csv'))
    return df


def plot_type_histogram(df: pd.DataFrame):
    """
    Plot the histogram of number of objects per type
    """
    plt.figure(figsize=(10, 5))
    sns.countplot(y='type', data=df,
                  order = df['type'].value_counts().index,
                  palette='rocket')
    plt.title('Number of objects per type')
    # plt.savefig(os.path.join(root_dir, 'type_histogram.png'))
    plt.show()
    return

def generate_dataset_split(df: pd.DataFrame):
    # Split into train, unseen instance, unseen class

    # Unseen instance: 10 instances, 1 instance from each classes with >= 3 instances
    unseen_instance_classes = np.random.choice(df['type'].value_counts()[df['type'].value_counts() >= 3].index, 10, replace=False)
    unseen_instances = []
    for obj_type in unseen_instance_classes:
        obj_names = df[df['type'] == obj_type]['name'].values
        unseen_instances.extend(np.random.choice(obj_names, 1, replace=False))

    # Unseen class: 15 objects from the rest of the classes
    rest_classes = np.setdiff1d(df['type'].unique(), unseen_instance_classes)
    unseen_class_objs = []
    while len(unseen_class_objs) != 15:
        unseen_classes = np.random.choice(rest_classes, 7, replace=False)
        unseen_class_objs = df[df['type'].isin(unseen_classes)]['name'].values
    
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

def generate_unseen_type_split(df: pd.DataFrame, types=None):
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

def read_unstable_objects():
    unstable_objects = defaultdict(dict)
    unstable_object_files = [
        "unstable_objects_fixed_size.json",
        "unstable_objects_variable_size.json",
        "unstable_objects_variable_sizes_with_wall.json"]

    root_dir = "data/housekeep_all"
    for file in unstable_object_files:
        with open(os.path.join(root_dir, file), 'r') as f:
            data = json.load(f)
        
        for key, value in data.items():
            if value >= 12:
                unstable_objects[key][file] = value
    
    # Calculate the percentage of unstable poses for each object
    for obj, unstable_poses in unstable_objects.items():
        total_count = 0
        for mode, count in unstable_poses.items():
            total_count += count
        
        perc = total_count / 300
        unstable_objects[obj]['percentage'] = perc
        if perc >= 0.1:
            print(f'"{obj}",')
    
    # Export to json
    with open(os.path.join(root_dir, "all_unstable_objects.json"), 'w') as f:
        json.dump(unstable_objects, f, indent=2)
    
    print(f"Total unstable objects: {len(unstable_objects)}")
    
    return

def add_thick_splits():
    # Read the flat objects csv
    thick_obj_lookup = pd.read_csv(os.path.join(root_dir, 'flat_objects.csv'))
    thick_obj_lookup = thick_obj_lookup.set_index('name')
    
    # Read the current splits
    with open(os.path.join(root_dir, 'dataset_split.json'), 'r') as f:
        data = json.load(f)
    
    # Add the flat objects to the splits
    thick_splits = {}
    for split in data.keys():
        if split.endswith('_thick'):
            continue
        objects = data[split]
        thick_objects = []

        for obj in objects:
            if thick_obj_lookup['thick'][obj] == 'True':
                thick_objects.append(obj)
        
        split_name = f"{split}_thick"
        thick_splits[split_name] = thick_objects

        print(f"{split_name}: {len(thick_objects)}")
    
    # Export the new splits
    data.update(thick_splits)
    with open(os.path.join(root_dir, 'dataset_split.json'), 'w') as f:
        json.dump(data, f, indent=2)


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
    # df = generate_metadata()
    # plot_type_histogram(df)
    # generate_dataset_split(df)
    # generate_unseen_instance_split(df)
    # generate_unseen_type_split(df)
    # generated_mixed_split(df)
    add_thick_splits()
    # test_read()
    # read_unstable_objects()
