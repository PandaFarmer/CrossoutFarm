# https://stackoverflow.com/questions/69932895/count-total-number-of-pixels-for-each-color
#!/usr/bin/env python3
from PIL import Image
import numpy as np

IndicatorProperties = {"BlueAlliedTriangle.png": { "dominant_color": [ 69, 163, 255], "_count": 44, "div_dims": (2, 3)},
    "EscortMissionIndicator.png": { "dominant_color": [255, 150, 0], "_count": 630, "div_dims": (2,3)},
    "PurpleCrosshair.png": { "dominant_color": [191, 114, 252], "_count": 48, "div_dims": (2,2)},
    "RedChevron.png": { "dominant_color": [255,  50,  50], "_count": 80, "div_dims": (3,2)}}

def pixel_count_over_threshold(image, color, count):
    return color_count(image, color) > count

def color_count(image, color):
    # Make into Numpy array
    na = np.array(image)

    # Get colours and corresponding counts
    colours, counts = np.unique(na.reshape(-1,3), axis=0, return_counts=1)
    # print(f"colours {colours} \n counts {counts}")
    counts = list(counts)
    ColorIndex = counts.index(max(counts))
    _count = counts[ColorIndex]
    # print(f"ColorIndex: {ColorIndex}, value: {colours[ColorIndex]}, _count: {_count}")
    return _count

def dominant_color(fileName):
    """precursor function used for preprocessing and peeking into indicator info, not for use during main runtime"""
    im = Image.open(fileName).convert('RGB')
    na = np.array(im)

    # Get colours and corresponding counts
    colours, counts = np.unique(na.reshape(-1,3), axis=0, return_counts=1)
    # print(f"colours {colours} \n counts {counts}")
    counts = list(counts)
    ColorIndex = counts.index(max(counts))
    _count = counts[ColorIndex]
    dominant_color = colours[ColorIndex]
    print(f"Image {fileName}, dominant_color: {dominant_color}, _count: {_count}, Shape: {na.shape}")
    return colours[ColorIndex]


def file_pixel_count_over_threshold(fileName, color, count):
    # Open image and ensure RGB
    im = Image.open(fileName).convert('RGB')

    return pixel_count_over_threshold(im, color, count)

if __name__ == "__main__":
    dominant_color("Testing/CroppedIndicators/BlueAlliedTriangle.png")
    dominant_color("Testing/CroppedIndicators/EscortMissionIndicator.png")
    dominant_color("Testing/CroppedIndicators/PurpleCrosshair.png")
    dominant_color("Testing/CroppedIndicators/RedChevron.png")
    
# BlueAlliedTriangle.png, dominant_color: [ 69 163 255], _count: 44, Shape: (15, 16, 3)
# EscortMissionIndicator.png, dominant_color: [255 150   0], _count: 630, Shape: (94, 69, 3)
# PurpleCrosshair.png, dominant_color: [191 114 252], _count: 48, Shape: (30, 30, 3)
# RedChevron.png, dominant_color: [255  50  50], _count: 80, Shape: (16, 24, 3)