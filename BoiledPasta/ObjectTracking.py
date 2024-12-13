import numpy as np
from Logger import Logger
from Utilities import *
from dataclasses import dataclass
from typing import Tuple

@dataclass
class object_info(object):
    relative_location_xy: Tuple[int, int]
    location_on_screen: Tuple[int, int]

entity_highlights = []
obstacle_highlights = []
object_infos = {}

def read_speed():
    """reads in speed indicator"""
    im = screen_crop(900, 1670, 150, 150)
    return int(text_from_image(im))

def offset_movement(movement_vector):
    """provides a cartesian offset on entity movement based on estimated movement of vehicle"""
    Logger.INFO(module_name='ObjectTracking', msg='entity_minimap_info start')
    xmove, ymove = movement_vector
    shift_factor = .1
    return xmove*shift_factor, ymove*shift_factor
    
def offset_rotation(theta, entity_angles, entity_ranges, camera_position, ):
    """provides a rotational offset on entity movement based on estimated rotational change of vehicle"""
    Logger.INFO(module_name='ObjectTracking', msg='offset_rotation start')
    
    points_local_to_world(theta, entity_angles, entity_ranges, camera_position)
    return 

    
def label_entity_deltas(movement_offset, rotational_offset_matrix, ):
    """given vehicle offsets and tracked entities, estimates movement of entities"""
    Logger.INFO(module_name='ObjectTracking', msg='label_entity_deltas start')
    
def confirm_entity_destruction():
    """attempts to provide accurate confirmation of entity removal"""
    Logger.INFO(module_name='ObjectTracking', msg='confirm_entity_destruction start')
    
def update_entity_tracking():
    """applies decay function to entity tracking, removes any with score under threshold"""
    Logger.INFO(module_name='ObjectTracking', msg='update_entity_tracking start')