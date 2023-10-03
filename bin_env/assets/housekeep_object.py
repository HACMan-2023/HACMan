from collections import OrderedDict
import numpy as np
import pandas as pd

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string

import h5py
import tempfile
import json
import shutil
import os, random
from lxml import etree

from bin_env.assets.objects import MujocoMeshObject

curr_path = os.path.dirname(os.path.abspath(__file__))

class HousekeepSampler:
    def __init__(self, use_full_set=False, object_scale_range=None, object_size_limit=[0.05, 0.05], **kwargs):
        if use_full_set:
            self.housekeep_root = os.path.join(curr_path, 'housekeep_all')
        else:
            self.housekeep_root = os.path.join(curr_path, 'housekeep')
        
        self.objects_root = os.path.join(self.housekeep_root, 'models_processed')
        self.object_info_root = os.path.join(self.housekeep_root, 'object_info')
        self.texture_root = os.path.join(curr_path, 'textures')
            
        self.object_set, self.all_object_info = self.load_all_objects(**kwargs)
        self.object_scale_range = object_scale_range
        self.object_size_limit = object_size_limit


    def load_all_objects(self, object_types=None, object_name=None, object_split=None, **kwargs):
        object_df = pd.read_csv(os.path.join(self.housekeep_root, 'metainfo.csv'))
        object_set = []
        all_object_info = {}
        
        # Found the set of specified objects
        if object_name is None:            
            # Sample from a split configuration
            if object_split is not None and (object_split != 'no_split'):
                with open(os.path.join(self.housekeep_root, 'dataset_split.json')) as f:
                    split = json.load(f)
                object_set = split[object_split]
            
            # Sample from specific types
            elif object_types is not None:
                object_set = object_df[object_df['type'].isin(object_types)]['name'].tolist()
            
            # Sample from all objects
            else:
                object_set = object_df['name'].values.tolist()
            
        else:
            assert object_name in object_df['name'].values.tolist(), f"Specified object {object_name} does not exist."
            object_set = [object_name]
            
        assert len(object_set) > 0, "No objects found in the specified set."
        
        # Load all object info
        for object_name in object_set:
            info_path = os.path.join(self.object_info_root, object_name+'.h5')
            assert os.path.exists(info_path), f"Specified object {object_name} does not exist."
            all_object_info[object_name] = h5py.File(info_path, "r")

        print(f"Loaded {len(object_set)} objects from {self.housekeep_root}.")
        
        return object_set, all_object_info
    

    def set_object(self, object_name):
        assert object_name in self.all_object_info.keys(), f"Specified object {object_name} does not exist."
        self.object_set = [object_name]
    

    def get_object_set(self):
        return self.object_set

    def generate_housekeep_object(self):
        object_info = self.sample_housekeep_object()

        obj = MujocoMeshObject(object_info=object_info, 
                            name="object", 
                            stl_dir=self.objects_root,
                            texture_dir=self.texture_root, 
                            joints=[dict(type="free", damping="0.0005")], 
                            obj_type="all", 
                            duplicate_collision_geoms=True, 
                            size_limit=self.object_size_limit,
                            object_scale_range=self.object_scale_range)
        
        obj_target = MujocoMeshObject(object_info=object_info, 
                                    name="object_target", 
                                    stl_dir=self.objects_root, 
                                    texture_dir=self.texture_root, 
                                    joints=None, 
                                    obj_type="visual",
                                    duplicate_collision_geoms=True, 
                                    size_limit=self.object_size_limit,
                                    object_scale_range=self.object_scale_range,
                                    object_scale=obj.scale)
        
        return obj, obj_target

    def sample_housekeep_object(self):        
        object_name = np.random.choice(self.object_set).item()
        object_info = self.all_object_info[object_name]

        return object_info

# class HousekeepObject(MujocoXMLObject):
#     """
#     Loading Acronym Objects and Grasps
#     """
#     def __init__(self, name=None, joints=None, object_info=None, size=None, density=None, obj_type='all', pos_offset=None):
#         # filename = object_info.filename.split('/')[-1].split('_')[1] + ".h5"
#         # additional_data = h5py.File(os.path.join(ACRONYM_OBJECT_INFO_ROOT, filename), "r")
#         self.temp_dir = tempfile.TemporaryDirectory(dir=HOUSEKEEP_OBJECTS_ROOT)
#         self.origin = object_info['origin'][()]
#         self.original_size = object_info['size'][()]
#         self.size = size # np.array([0.05, 0.05, 0.02])
#         self.density = density
#         self.pos_offset = pos_offset
#         self.mesh_name = object_info.filename.split('/')[-1].rstrip('.h5')
        
#         self.xml_path = self.parse_object_info(object_info)
#         super().__init__(self.xml_path, name=name, joints=joints, obj_type=obj_type, duplicate_collision_geoms=True)

#     def __del__(self):
#         self.temp_dir.cleanup()
#         try:
#             if os.path.exists(self.temp_dir.name):
#                 shutil.rmtree(self.temp_dir.name)
#         finally:
#             del self.temp_dir

#     def parse_object_info(self, data):
#         orig_size_norm = np.max(self.original_size)
#         # orig_size_norm = np.sqrt(np.sum(self.original_size ** 2))
#         # orig_size_norm = np.sum(self.original_size)
#         object_scale = np.ones(3) / orig_size_norm * self.size
#         object_com = self.origin * object_scale
#         if self.pos_offset is not None:
#             object_com += self.pos_offset

#         # Save them for other process to read
#         self.scale = object_scale
#         self.scaled_origin = object_com
#         self.scaled_size = self.original_size * object_scale

#         # # Using the scale from acronym
#         # object_scale = np.ones(3) * data['object/scale'][()]
#         # # object_com = self.origin * data['object/scale'][()]
#         # object_com = data["object/com"][()]

#         object_quat = np.array([1, 0, 0, 0])
#         # # Change the orientation according to the shape in the order of x>y>z # BUG
#         # size = self.size.copy()
#         # original_size = self.original_size.copy()
#         # print(original_size)
#         # if original_size[1] < original_size[2]:
#         #     object_quat = quat_mul(euler2quat(np.array([np.pi/2, 0, 0])), object_quat)
#         #     original_size[2], original_size[1] = original_size[1], original_size[2]
#         #     size[2], size[1] = size[1], size[2]
#         # if original_size[0] < original_size[2]:
#         #     object_quat = quat_mul(euler2quat(np.array([0, np.pi/2, 0])), object_quat)
#         #     object_quat = quat_mul(euler2quat(np.array([0, 0, np.pi/2])), object_quat)
#         #     original_size[0], original_size[1], original_size[2] = original_size[1], original_size[2], original_size[0]
#         #     size[0], size[1], size[2] = size[1], size[2], size[0]
#         # elif original_size[0] < original_size[1]:
#         #     object_quat = quat_mul(euler2quat(np.array([0, 0, np.pi/2])), object_quat)
#         #     original_size[0], original_size[1] = original_size[1], original_size[0]
#         #     size[0], size[1] = size[1], size[0]
#         # print(original_size)
#         # object_scale = np.ones(3) / self.original_size * size
#         # object_com = self.origin / self.original_size * size
#         # object_com = quat_rot_vec_arr(object_quat, object_com)

#         filename = data.filename.split('/')[-1]
#         object_type = filename.split('_')[0]
#         object_id = filename.split('.')[0]
#         object_texture = os.path.join(TEXTURE_ROOT, "ceramic.png")
#         # object_mass = str(data["object/mass"][()])
#         object_density = str(self.density) #str(data["object/density"][()])
#         # object_friction = str(data["object/friction"][()])
#         object_friction = "1.0"
#         xml_path = self.generate_xml(object_id, object_type, object_scale, object_texture, 
#                                      object_density, object_friction, object_com, object_quat)
#         return xml_path

#     def generate_xml(self, object_id, object_type, object_scale, object_texture, object_density,
#                      object_friction, object_com, object_quat):
#         object_xml = object_id + '.xml'
#         # object_xml_path = os.path.join(HOUSEKEEP_OBJECTS_ROOT, object_xml)
#         object_xml_path = os.path.join(self.temp_dir.name, object_xml)
#         if True: # not os.path.exists(object_xml_path): # Always generate new files because scale and mass might change
#             root = etree.Element("mujoco")
#             root.set("model", object_id)

#             asset = etree.SubElement(root, "asset")
#             mesh = etree.SubElement(asset, "mesh")
#             mesh.set("file", "../meshes/" + object_id + ".stl")
#             mesh.set("name", object_type + "_mesh")
#             mesh.set("scale", str(object_scale[0]) + " " + str(object_scale[1]) + " " + str(object_scale[2]))
            
#             texture = etree.SubElement(asset, "texture")
#             texture.set("type", "cube")
#             texture.set("file", object_texture)
#             texture.set("rgb1", "1 1 1")
#             texture.set("name", "tex-" + object_type)
#             material = etree.SubElement(asset, "material")
#             material.set("name", object_type)
#             material.set("reflectance", "0.5")
#             material.set("texrepeat", "3 3")
#             material.set("texture", "tex-" + object_type)

#             # TODO: check XML (such as mass values)
#             worldbody = etree.SubElement(root, "worldbody")
#             body = etree.SubElement(worldbody, "body")
#             collision = etree.SubElement(body, "body")
#             collision.set("name", "object")
#             geom = etree.SubElement(collision, "geom")
#             geom.set("pos", str(-object_com[0])+" "+str(-object_com[1])+" "+str(-object_com[2]))
#             # geom.set("pos", str(object_com[0])+" "+str(object_com[1])+" "+str(-object_com[2]))
#             # geom.set("pos", str(object_com[0])+" "+str(object_com[1])+" "+str(0))
#             geom.set("quat", str(object_quat[0])+" "+str(object_quat[1])+" "+str(object_quat[2])+" "+str(object_quat[3]))
#             # geom.set("pos", "0 0 0")
#             geom.set("mesh", object_type + "_mesh")
#             geom.set("type", "mesh")
#             geom.set("density", object_density)

#             # Mujoco default
#             # geom.set("solimp", "0.9 0.95 0.001")
#             # geom.set("solref", "0.02 1")
#             # geom.set("friction", object_friction + " " + object_friction + " " + object_friction)
            
#             # Robosuite xml object settings
#             # geom.set("solimp", "0.998 0.998 0.001")
#             # geom.set("solref", "0.001 1")
#             # geom.set("density", "50")
#             # geom.set("friction", "0.95 0.3 0.1")
            
#             geom.set("group", "0")
#             geom.set("condim", "4")
#             geom.set("material", object_type)

#             # visual = etree.SubElement(body, "body")
#             # visual.set("name", "visual")
#             # geom1 = etree.SubElement(visual, "geom")
#             # geom1.set("pos", "0 0 0")
#             # geom1.set("mesh", object_type + "_mesh")
#             # geom1.set("type", "mesh")
#             # geom1.set("material", object_type)
#             # geom1.set("conaffinity", "0")
#             # geom1.set("contype", "0")
#             # geom1.set("group", "0")
#             # geom1.set("mass", "0.1")

#             # geom2 = etree.SubElement(visual, "geom")
#             # geom2.set("pos", "0 0 0")
#             # geom2.set("mesh", object_type + "_mesh")
#             # geom2.set("type", "mesh")
#             # geom2.set("material", object_type)
#             # geom2.set("conaffinity", "0")
#             # geom2.set("contype", "0")
#             # geom2.set("group", "1")
#             # geom2.set("mass", "0.1")

#             # TODO: set the sites properly
#             bottom_site = etree.SubElement(body, "site")
#             bottom_site.set("rgba", "0 0 0 0")
#             bottom_site.set("size", "0.005")
#             bottom_site.set("pos", "0 0 -0.05")
#             bottom_site.set("name", "bottom_site")
#             top_site = etree.SubElement(body, "site")
#             top_site.set("rgba", "0 0 0 0")
#             top_site.set("size", "0.005")
#             top_site.set("pos", "0 0 0.03")
#             top_site.set("name", "top_site")
#             horizontal_radius_site = etree.SubElement(body, "site")
#             horizontal_radius_site.set("rgba", "0 0 0 0")
#             horizontal_radius_site.set("size", "0.005")
#             horizontal_radius_site.set("pos", "0.02 0.015 0")
#             horizontal_radius_site.set("name", "horizontal_radius_site")

#             et = etree.ElementTree(root)
#             # if object_xml_path is not None and os.path.exists(object_xml_path):
#             #     os.remove(object_xml_path)
#             et.write(object_xml_path, pretty_print=True)
#         return object_xml_path
    

if __name__ == '__main__':
    sampler = HousekeepSampler(object_types='all', object_name=None, object_split=None, object_eval_set=False)
    for _ in range(15):
        obj, target_obj = sampler.generate_housekeep_object()
        print(obj.mesh_name)