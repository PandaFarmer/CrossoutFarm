# https://stackoverflow.com/questions/72900640/divide-an-image-into-radial-sections-and-select-values-in-an-individual-section

import cv2
import math
import numpy as np

h, w   = 600, 1200   # image height and width
cx, cy = 200, 300    # (x,y) coordinates of circle centre
N      = 16          # number of slices in our pie
l      = h + w       # length of radial lines - larger than necessary

# Create each sector in white on black background
for sector in range(N):
    startAngle = sector * 360/N
    endAngle   = startAngle + 360/N
    x1 = cx + l * math.sin(math.radians(startAngle))
    y1 = cy - l * math.cos(math.radians(startAngle))
    x2 = cx + l * math.sin(math.radians(endAngle))
    y2 = cy - l * math.cos(math.radians(endAngle))
    vertices = [(cy, cx), (y1, x1), (y2, x2)]
    print(f'DEBUG: sector={sector}, startAngle={startAngle}, endAngle={endAngle}')
    # Make empty black canvas
    im = np.zeros((h,w), np.uint8)
    # Draw this pie slice in white
    cv2.fillPoly(im, np.array([vertices],'int32'), 255)
    cv2.imwrite(f'DEBUG-{sector:03d}.png', im)
    cv2.imshow('title',im)
    cv2.waitKey(0)