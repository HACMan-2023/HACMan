import os
from robosuite.models.grippers.gripper_model import GripperModel


class CustomizedRobotiq85Gripper(GripperModel):
    """
    6-DoF Robotiq gripper with festo fingertip.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """
    def __init__(self, idn=0):
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/grippers', 'robotiq_gripper_85.xml')
        super().__init__(xml_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision"
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision"
            ],
            "left_fingerpad": [
                "left_fingerpad_collision"
            ],
            "right_fingerpad": [
                "right_fingerpad_collision"
            ],
        }
    
    @property
    def dof(self):
        return 0