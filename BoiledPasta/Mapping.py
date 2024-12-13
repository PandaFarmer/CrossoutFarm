import py_trees
from py_trees.behaviour import Behaviour
import pyautogui
from Logger import Logger
from Utilities import *
from ColorCount import IndicatorProperties
import numpy as np


class Mapping(Behaviour):
    def __init__(self, name):
        super(Mapping, self).__init__(name)
        self.hasrun = False
        self.name = name
        self.large_map_raw = None
        self.minimap_raw = None
        self.radial_slices = None
        self.prev_camera_azimuth = None
        self.camera_azimuth = None

    def update(self):
        if self.hasrun:
            return py_trees.common.Status.SUCCESS
        if self.name == 'check_large_map':
            return self.check_large_map()
        if self.name == 'check_minimap':
            return self.check_minimap()
        if self.name == 'update_camera_azimuth':
            return self.update_camera_azimuth()
        if self.name == 'camera_azimuth_delta':
            return self.camera_azimuth_delta()
        if self.name == 'entity_minimap_info':
            return self.update_entity_estimates()

        return py_trees.common.Status.RUNNING

    def check_large_map(self):
        """prompts tab usage and scans and stores large map info"""
        Logger.INFO(module_name='Mapping', msg='check_large_map start')
        large_map_width, large_map_height = 780, 980
        top, left = (SCREEN_WIDTH -
                     large_map_width)//2, (SCREEN_HEIGHT - large_map_height)//2
        with pyautogui.hold('\t'):
            self.large_map_raw = screen_crop(
                top=top, left=left, width=large_map_width, height=large_map_height)

    def check_minimap(self):
        """using cropped image of minimap, stores info"""
        Logger.INFO(module_name='Mapping', msg='check_minimap start')
        h, w = 200, 200
        top, left = SCREEN_WIDTH - h, SCREEN_HEIGHT - w
        center = SCREEN_WIDTH - h//2, SCREEN_HEIGHT - w//2
        self.minimap_raw = screen_crop(top=top, left=left, height=h, width=w)
        self.radial_slices = radial_slices(
            self.minimap_raw, h, w, center, num_slices=16)

    def update_camera_azimuth(self):
        """estimates camera azimuth based on minimap"""
        Logger.INFO(module_name='Mapping', msg='update_camera_azimuth start')
        num_slices = len(radial_slices)
        slice_angle = 360/num_slices
        b_threshold = 12
        def bright_fn(
            r, g, b): return r < b_threshold and g < b_threshold and b < b_threshold
        # int(r < b_threshold) + (g < b_threshold) + int(b < b_threshold) > 2
        bright_scores = [np.apply_along_axis(
            bright_fn, 0, radial_slice).pop() for radial_slice in radial_slices]
        # find circular median
        FOV_ARC = 90
        CONSECUTIVE_SLICES_ON_FOV_ARC = int(360/FOV_ARC*1/slice_angle)
        median_index = 0
        num_consecutive = 0
        total_traversal = 0
        BRIGHTNESS_SCORE_THRESHOLD = 2
        while total_traversal < num_slices*1.5:
            if num_consecutive >= CONSECUTIVE_SLICES_ON_FOV_ARC:
                self.camera_azimuth = median_index*slice_angle - FOV_ARC//2
                return
            if bright_scores[median_index] > BRIGHTNESS_SCORE_THRESHOLD:
                num_consecutive += 1
            else:
                num_consecutive = 0
            median_index += 1
            median_index %= num_slices
            total_traversal += 1

        raise ValueError(
            "Could not update camera azimuth with brightness values")

    def camera_azimuth_delta(self):
        """estimate camera azimuth change"""
        Logger.INFO(module_name='Mapping', msg='camera_azimuth_delta start')
        if self.camera_azimuth and self.prev_camera_azimuth:
            return self.camera_azimuth - self.prev_camera_azimuth

    def update_entity_estimates(self):
        """loose information on entity position given minimap info"""

        Logger.INFO(module_name='Mapping', msg='entity_minimap_info start')
        blue = IndicatorProperties["BlueAlliedTriangle.png"]["dominant_color"]
        red = IndicatorProperties["EscortMissionIndicator.png"]["dominant_color"]
        purple = IndicatorProperties["PurpleCrosshair.png"]["dominant_color"]

        mean_blue = mean_location_of_color_pixels(self.minimap_raw, blue)
        mean_red = mean_location_of_color_pixels(self.minimap_raw, red)
        mean_purple = mean_location_of_color_pixels(self.minimap_raw, purple)
        return mean_2d([mean_blue, mean_red, mean_purple])
