# https://claude.ai/chat/3d9457da-72f6-41b1-a72f-4527084d7bd0
import numpy as np
from sklearn.linear_model import LinearRegression
import time

def linear_regression_numpy(points):
    """
    Fast linear regression using numpy's matrix operations.
    
    Parameters:
    points (np.ndarray): Binary array of shape (n, 2) containing x, y coordinates
    
    Returns:
    tuple: (slope, intercept)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Compute slope using vectorized operations
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    # Calculate slope and intercept
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept