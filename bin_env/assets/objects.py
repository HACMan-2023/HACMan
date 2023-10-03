import os
import numpy as np
import random
import logging
import xml.etree.ElementTree as ET
from robosuite.models.objects.objects import MujocoXMLObject, GEOM_GROUPS
    
    
class MujocoTreeObject(MujocoXMLObject):
    """
    Initialize a mujoco object without creating the xml file
    """
    def __init__(self, tree, name, root_dir=None, joints="default", obj_type="all", duplicate_collision_geoms=True):
        self.init_mujocoxml(tree, root_dir)
        self.init_mujocoxmlobject(name, joints, obj_type, duplicate_collision_geoms)

    def init_mujocoxml(self, tree, folder=None):
        """
        __init__ from MujocoXML
        """
        # Root dir to be used for any relative path in the xml file
        self.folder = folder
        
        self.tree = tree
        
        self.root = self.tree.getroot()
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.sensor = self.create_default_element("sensor")
        self.asset = self.create_default_element("asset")
        self.tendon = self.create_default_element("tendon")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")

        # Parse any default classes and replace them inline
        default = self.create_default_element("default")
        default_classes = self._get_default_classes(default)
        self._replace_defaults_inline(default_dic=default_classes)

        # Remove original default classes
        self.root.remove(default)

        self.resolve_asset_dependency()
        
    def init_mujocoxmlobject(self, name, joints="default", obj_type="all", duplicate_collision_geoms=True):
        """
        __init__ from MujocoXMLObject
        """
        
        # Set obj type and duplicate args
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms

        # Set name
        self._name = name

        # joints for this object
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Make sure all joints have names!
        for i, joint_spec in enumerate(self.joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(i)

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # Extract the appropriate private attributes for this
        self._get_object_properties()


class MujocoMeshObject(MujocoTreeObject):
    def __init__(self, object_info, name, stl_dir=None, texture_dir=None, joints="default", obj_type="all", duplicate_collision_geoms=True, 
                 size_limit=None, xml_callback=None, object_scale_range=(0.8, 1.2), object_scale=None):
        self.size_limit = np.array(size_limit) if size_limit is not None else None
        self.obj_type = obj_type
        self.stl_dir = stl_dir
        if object_scale_range is None:
            self.scales = np.array([1.0])
        else:
            self.scales = np.arange(object_scale_range[0], object_scale_range[1]+0.001, 0.1).round(2)

        self.mesh_name = object_info.filename.split('/')[-1].rstrip('.h5')
        self.texture_dir = texture_dir
        self.xml_callback = xml_callback # Additional changes to the xml file

        et = self.parse_object_info(object_info, object_scale=object_scale)
        super().__init__(et, name=name, root_dir=stl_dir, joints=joints, obj_type=obj_type, duplicate_collision_geoms=duplicate_collision_geoms)

    def parse_object_info(self, object_info, object_scale=None):
        origin = object_info['origin'][()]
        original_size = object_info['size'][()]
        filename = object_info.filename.split('/')[-1]
        object_type = filename.split('_')[0]
        object_id = filename.split('.')[0]
        object_texture = os.path.join(self.texture_dir, "color.png")
        
        if object_scale is not None:
            pass

        elif self.size_limit is None:
            random_scale = random.choice(self.scales)
            object_scale = np.ones(3) * random_scale

        elif self.size_limit.shape == (2,):
            # Limit the size of the object while keeping the shape
            orig_size_norm = np.max(original_size)
            object_scale = np.ones(3)
            if orig_size_norm > self.size_limit[1]:
                object_scale *= self.size_limit[1] / orig_size_norm
            elif orig_size_norm < self.size_limit[0]:
                object_scale *= self.size_limit[0] / orig_size_norm

            # Multiplies the scale by a random factor
            random_scale = random.choice(self.scales)
            object_scale *= random_scale

        # elif self.size_limit.shape == (3,):
        #     # Scale the object in x,y,z dimension separately
        #     object_scale = np.ones(3) / original_size * self.size_limit
        #     object_com = origin * object_scale
        #     object_dimension = original_size * object_scale

        else:
            raise NotImplementedError
        
        object_com = origin * object_scale
        object_dimension = original_size * object_scale

        # Save them for other process to read
        self.scale = object_scale
        self.scaled_origin = object_com
        self.scaled_size = object_dimension

        # Create Element Tree
        root = ET.Element("mujoco")
        root.set("model", object_id)

        asset = ET.SubElement(root, "asset")
        mesh = ET.SubElement(asset, "mesh")
        mesh.set("file", self.stl_dir + "/meshes/" + object_id + ".stl")
        mesh.set("name", object_type + "_mesh")
        mesh.set("scale", str(object_scale[0]) + " " + str(object_scale[1]) + " " + str(object_scale[2]))
        
        texture = ET.SubElement(asset, "texture")
        texture.set("file", object_texture)
        texture.set("type", "2d")
        texture.set("name", "tex-" + object_type)
        
        material = ET.SubElement(asset, "material")
        material.set("name", object_type)
        material.set("reflectance", "0.7")
        material.set("texrepeat", "15 15")
        material.set("texture", "tex-" + object_type)
        material.set("texuniform", "true")

        worldbody = ET.SubElement(root, "worldbody")
        body = ET.SubElement(worldbody, "body")
        collision = ET.SubElement(body, "body")
        collision.set("name", "object")
        geom = ET.SubElement(collision, "geom")
        geom.set("pos", str(-object_com[0])+" "+str(-object_com[1])+" "+str(-object_com[2]))
        geom.set("mesh", object_type + "_mesh")
        geom.set("type", "mesh")

        if self.obj_type == 'visual':
            geom.set("rgba", "0.8 0.8 0.8 0.3")
            geom.set("conaffinity", "0")
            geom.set("contype", "0")
            geom.set("group", "1")
            geom.set("mass", "0.0001")
            geom.set("material", object_type)
        else:
            # Robosuite xml object settings
            geom.set("solimp", "0.998 0.998 0.001")
            geom.set("solref", "0.001 1")
            geom.set("density", "50")
            geom.set("friction", "0.95 0.3 0.1") # Currently overwritten by table friction            
            geom.set("group", "0")
            geom.set("condim", "4")
            geom.set("material", object_type)

        bottom_site = ET.SubElement(body, "site")
        bottom_site.set("rgba", "1 0 0 1")
        bottom_site.set("size", "0.005")
        bottom_site.set("pos", "0 0 " + str(object_dimension[2]/2-0.1))
        bottom_site.set("name", "bottom_site")
        top_site = ET.SubElement(body, "site")
        top_site.set("rgba", "1 0 0 1")
        top_site.set("size", "0.005")
        top_site.set("pos", "0 0 " + str(-object_dimension[2]/2-0.1))
        top_site.set("name", "top_site")
        horizontal_radius_site = ET.SubElement(body, "site")
        horizontal_radius_site.set("rgba", "1 1 0 1")
        horizontal_radius_site.set("size", "0.005")
        horizontal_radius_site.set("pos", str(object_dimension[0]) + " " + str(object_dimension[1])+" 0")
        horizontal_radius_site.set("name", "horizontal_radius_site")
        
        if self.xml_callback is not None:
            # Additional changes to the xml file
            self.xml_callback(globals(), locals())

        et = ET.ElementTree(root)
        return et
