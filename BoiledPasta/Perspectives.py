
import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt
from Utilities import tile_split, apply_sobel_filter, interpolate_line_from_image


def orthogonal_lines_from_perspective(image):
    """splits filtered image into grid and interpolates orthogonal lines"""
    image = image.convert('L')
    image = apply_sobel_filter(image)
    M, N = .05*image.shape
    tile_grid = tile_split(image, int(M), int(N))

    # howto convolve with step size == window_size to aggregate pixel values?
    # https://www.python-engineer.com/posts/image-thresholding/
    # https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
    
    Vinterpolate_line_from_image = np.vectorize(interpolate_line_from_image)
    slopes = Vinterpolate_line_from_image(tile_grid)
    return slopes

def vanishing_points_from_slopes(slopes, initial_image_dim):
    """interpolates vanishing point(s) on camera screen from orthogonal lines"""
    slopeX, slopeY = slopes.shape
    initial_image_dimX, initial_image_dimY = initial_image_dim.shape
    x_scale, y_scale = initial_image_dimX/slopeX, initial_image_dimY/slopeY
    rv = []
    
    #TODO convolution? with what kernel? take make score per row? then scale up for return coords?
    #or go from top and bottom rows, track slope changes and adjacent slopes changes, 
    #sobely?
    
            
    return 

def vanishing_points_from_filtered_image():
    """interpolates vanishing point(s) on camera screen from downscaled image"""
    kernel = [[1,  1,  1],
            [1, -8,  1],
            [1,  1, 1]]
    
def horizon_lines(image, slopes):
    """interpolates horizon line(s) from orthogonal lines and additional heuristics"""
    

def entity_info(bounding_boxes):
    """
    given a list of bounding boxes in format (x1, y1, x2, y2), 
    interpolates entity distance and position from camera
    returns as dictionary with bounding boxes as keys
    """
    
def obstacle_info():
    """
    returns a list of obstacles relative to camera position
    """