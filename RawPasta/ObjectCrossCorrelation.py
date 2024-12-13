# https://claude.ai/chat/30b0e915-39b7-4c49-81b4-d453b52c37b2

import numpy as np
from scipy import signal
import cv2

def track_motion(img1, img2, roi=None, subpixel=True):
    """
    Track motion between two images using cross-correlation.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image (reference)
    img2 : numpy.ndarray
        Second image (target)
    roi : tuple, optional
        Region of interest in format (x, y, width, height)
    subpixel : bool, optional
        Whether to use subpixel interpolation for better accuracy
        
    Returns:
    --------
    tuple
        (dx, dy) displacement vector in pixels
    """
    # Convert images to grayscale if they're in color
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Extract ROI if specified
    if roi is not None:
        x, y, w, h = roi
        img1 = img1[y:y+h, x:x+w]
        img2 = img2[y:y+h, x:x+w]
    
    # Normalize images to reduce effects of lighting changes
    img1 = (img1 - np.mean(img1)) / np.std(img1)
    img2 = (img2 - np.mean(img2)) / np.std(img2)
    
    # Compute cross-correlation
    correlation = signal.correlate2d(img1, img2, mode='same', boundary='symm')
    
    # Find the peak correlation
    y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Calculate displacement from center
    center_y, center_x = correlation.shape[0] // 2, correlation.shape[1] // 2
    dy = y_peak - center_y
    dx = x_peak - center_x
    
    if subpixel:
        # Refine displacement using subpixel interpolation
        if 0 < y_peak < correlation.shape[0]-1 and 0 < x_peak < correlation.shape[1]-1:
            # Fit parabola to peak and neighboring points
            y_neighbors = correlation[y_peak-1:y_peak+2, x_peak]
            x_neighbors = correlation[y_peak, x_peak-1:x_peak+2]
            
            # Subpixel refinement using parabolic fit
            dx_subpixel = (x_neighbors[0] - x_neighbors[2]) / (2 * (x_neighbors[0] + x_neighbors[2] - 2*x_neighbors[1]))
            dy_subpixel = (y_neighbors[0] - y_neighbors[2]) / (2 * (y_neighbors[0] + y_neighbors[2] - 2*y_neighbors[1]))
            
            dx += dx_subpixel
            dy += dy_subpixel
    
    return dx, dy

def estimate_motion_confidence(correlation, peak_pos):
    """
    Estimate the confidence of the motion tracking result.
    
    Parameters:
    -----------
    correlation : numpy.ndarray
        Cross-correlation matrix
    peak_pos : tuple
        Position of correlation peak (y, x)
        
    Returns:
    --------
    float
        Confidence score between 0 and 1
    """
    y_peak, x_peak = peak_pos
    peak_value = correlation[y_peak, x_peak]
    
    # Calculate mean and std of correlation excluding peak region
    mask = np.ones_like(correlation)
    mask[max(0, y_peak-1):min(correlation.shape[0], y_peak+2),
         max(0, x_peak-1):min(correlation.shape[1], x_peak+2)] = 0
    
    background_mean = np.mean(correlation[mask == 1])
    background_std = np.std(correlation[mask == 1])
    
    # Calculate peak sharpness (higher is better)
    peak_sharpness = (peak_value - background_mean) / (background_std + 1e-10)
    
    # Convert to confidence score between 0 and 1
    confidence = 1 - np.exp(-peak_sharpness / 10)
    return min(1.0, max(0.0, confidence))