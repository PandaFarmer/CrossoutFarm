import cv2
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class SceneFeatures:
    horizon_line: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1,y1), (x2,y2))
    vanishing_points: List[Tuple[int, int]]  # [(x,y), ...]
    obstacles: List[np.ndarray]  # [contour, ...]

def analyze_scene(frame: np.ndarray) -> SceneFeatures:
    """
    Analyze a scene to find horizon line, vanishing points, and obstacles.
    Uses classical computer vision techniques without neural networks.
    
    Args:
        frame: Input image as numpy array (BGR format)
    Returns:
        SceneFeatures object containing detected elements
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Find horizon line using edge detection and Hough transform
    def find_horizon(img):
        # Apply edge detection
        edges = cv2.Canny(img, 50, 150)
        
        # Use probabilistic Hough transform to find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return ((0, 0), (frame.shape[1], frame.shape[0]//2))
        
        # Filter for mostly horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
            if angle < 20 or angle > 160:
                horizontal_lines.append(line[0])
        
        if not horizontal_lines:
            return ((0, 0), (frame.shape[1], frame.shape[0]//2))
        
        # Take the median y-value as horizon
        median_y = np.median([line[1] for line in horizontal_lines])
        return ((0, int(median_y)), (frame.shape[1], int(median_y)))
    
    # Step 2: Find vanishing points using line intersection
    def find_vanishing_points(img):
        # Edge detection and Hough lines
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        vanishing_points = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # Calculate intersection
                denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                if denom == 0:  # parallel lines
                    continue
                    
                px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
                
                # Filter points within reasonable range
                if (0 <= px <= frame.shape[1] and 
                    0 <= py <= frame.shape[0]):
                    vanishing_points.append((int(px), int(py)))
        
        # Cluster nearby points and return centroids
        if vanishing_points:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            points = np.float32(vanishing_points)
            _, labels, centers = cv2.kmeans(points, min(3, len(points)), 
                                          None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            return [(int(x), int(y)) for x, y in centers]
        return []
    
    # Step 3: Detect obstacles using contour detection
    def find_obstacles(img):
        # Apply background subtraction / thresholding
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        min_area = frame.shape[0] * frame.shape[1] * 0.01  # 1% of image area
        obstacles = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return obstacles
    
    # Execute pipeline
    horizon = find_horizon(gray)
    vanishing = find_vanishing_points(gray)
    obstacles = find_obstacles(gray)
    
    return SceneFeatures(horizon, vanishing, obstacles)

def visualize_results(frame: np.ndarray, features: SceneFeatures) -> np.ndarray:
    """
    Visualize detected features on the input frame
    """
    vis = frame.copy()
    
    # Draw horizon line
    cv2.line(vis, features.horizon_line[0], features.horizon_line[1], 
             (0, 255, 0), 2)
    
    # Draw vanishing points
    for vp in features.vanishing_points:
        cv2.circle(vis, vp, 5, (0, 0, 255), -1)
    
    # Draw obstacles
    cv2.drawContours(vis, features.obstacles, -1, (255, 0, 0), 2)
    
    return vis

# Example usage
def process_video_stream(video_source=0):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze frame
        features = analyze_scene(frame)
        
        # Visualize results
        vis = visualize_results(frame, features)
        
        # Display
        cv2.imshow('Scene Analysis', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()