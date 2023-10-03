import os
import open3d as o3d
import math
import json

from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'models_processed', 'meshes') 
IMAGE_DIR = os.path.join(ROOT_DIR, 'images')

def vis_object(obj_name):
    """Visualize an object mesh and save it to a PNG file"""
    obj_file = os.path.join(MODEL_DIR, obj_name + '.stl')
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()

    # Create a visualization settings object and set the desired view angle
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=400, height=400, visible=True)
    vis.add_geometry(mesh)
    vis.get_view_control().rotate(45.0, 45.0, 45.0)

    # Start the visualizer and allow user to interactively adjust the view
    vis.run()

    vis.update_renderer()
    vis.capture_screen_image(os.path.join(IMAGE_DIR, obj_name + '.png'), do_render=True)
    vis.destroy_window()

def generate_images():
    """Generate PNG images for all STL files in the model folder"""
    print("Generating images...")
    for obj_name in tqdm(os.listdir(MODEL_DIR)):
        if obj_name.endswith('.stl'):
            obj_name = os.path.splitext(obj_name)[0]
            vis_object(obj_name)

def align_images(split=None):
    """Align all PNG images into a single picture"""
    images = []
    split_file = os.path.join(ROOT_DIR, "dataset_split.json")
    with open(split_file, 'r') as f:
        data = json.load(f)
        if split is not None:
            object_list = data[split]
        else:
            object_list = data["train"] + data["unseen_instance"] + data["unseen_class"]

    for obj_name in object_list:
        image_path = os.path.join(IMAGE_DIR, obj_name + '.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)

    if images:
        # Calculate the number of rows and columns based on the aspect ratio
        num_images = len(images)
        aspect_ratio = 16 / 9
        num_columns = int(math.sqrt(num_images / aspect_ratio))
        num_rows = int(math.ceil(num_images / num_columns))
        
        # Resize all images to the same height and calculate the width based on the aspect ratio
        target_height = 300
        target_width = int(target_height * aspect_ratio)
        resized_images = []
        for image in images:
            width, height = image.size
            ratio = target_height / height
            resized_image = image.resize((int(width * ratio), target_height))
            resized_images.append(resized_image)
        
        # Create a new image with the appropriate size
        total_width = num_columns * target_width
        total_height = num_rows * target_height
        result = Image.new('RGB', (total_width, total_height), color='white')
        
        # Paste the images into the new image
        index = 0
        for row in range(num_rows):
            for col in range(num_columns):
                if index >= num_images:
                    break
                x_offset = col * target_width
                y_offset = row * target_height
                result.paste(resized_images[index], (x_offset, y_offset))
                index += 1
        
        if split:
            img_name = 'objects_' + split + '.png'
        else:
            img_name = 'objects_all.png'
        result.save(os.path.join(ROOT_DIR, img_name))
    else:
        print('No PNG images found')

# generate_images()
# align_images()
align_images('train')
align_images('unseen_instance')
align_images('unseen_class')

# vis_object('box_Just_For_Men_Mustache_Beard_Brushin_Hair_Color_Gel_Kit_Jet_Black_M60_M')
# vis_object("cup_BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028_M")
# vis_object("pill_bottle_Beta_Glucan_M")
# vis_object("rubiks_cube_077_rubiks_cube_M")
# vis_object("pencil_case_Wishbone_Pencil_Case_M")

# vis_object("book_frl_apartment_book_03_M")
# vis_object("clamp_Pony_C_Clamp_1440_M")
# vis_object("tray_Threshold_Tray_Rectangle_Porcelain_M")
# vis_object("bowl_kitchen_set_kitchen_set_bowl_M")
# vis_object("handheld_game_console_Nintendo_2DS_Crimson_Red_M")
