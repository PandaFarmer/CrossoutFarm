import numpy as np
import scipy.ndimage as nd
import cv2
from mss import mss
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
import math

SCREEN_HEIGHT, SCREEN_WIDTH  = 1980, 1080

# https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python#6798042
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    
class ScreenRecorder(metaclass=Singleton):
    def __init__(self):
        self.sct = mss()
        
    def screen_crop(self, top=0, left=0, height=SCREEN_HEIGHT, width=SCREEN_WIDTH):
        bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}
        sct_img = self.sct.grab(bounding_box)
        # cv2.imshow('screen', np.array(sct_img))
        return sct_img
    
def screen_crop(top=0, left=0, height=SCREEN_HEIGHT, width=SCREEN_WIDTH):
    return ScreenRecorder.screen_crop(top, left, width, height)

def tile_split(im, M, N):
    tiles = [[im[x:x+M,y:y+N] for x in range(0,im.shape[0],M)] for y in range(0,im.shape[1],N)]
    # tiles = [[im[y:y+M,x:x+N] for x in range(0,im.shape[0],M)] for y in range(0,im.shape[1],N)]#NOT?
    
    #shaving off thin edge tiles
    _m, _n = M*.3, N*.3
    #TODO redo so grid propotions are always aligned as rectangle..? 
    tiles = [[tile for tile in tile_row if tile.shape[0] >= _m and tile.shape[1] >= _n] for tile_row in tiles]
    tiles = np.array([tile_row for tile_row in tiles if tile_row])
    return tiles

# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#15589825
def cropped_image(im, anchor):
    x, y, w, h = anchor
    cropped_img = im[y:y+h, x:x+w]
    return cropped_img

# https://stackoverflow.com/questions/36911877/cropping-circle-from-image-using-opencv-python#48709698
def cropped_image_circle(img, center, radius):
    img1 = img
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Create mask
    height,width = img.shape
    mask = np.zeros((height,width), np.uint8)
    # Draw on mask
    cv2.circle(mask,center,radius,(255,255,255),thickness=-1)

    # Copy that image using that mask
    masked_data = cv2.bitwise_and(img1, img1, mask=mask)

    # Apply Threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

    # Find Contour
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])

    # Crop masked_data
    crop = masked_data[y:y+h,x:x+w]
    
    return crop

def mean_location_of_color_pixels(im, color):
    x_s, y_s = [], []
    for x, y in np.ndindex(im.shape):
        if im[x, y] == color:
            x_s.append(x)
            y_s.append(y)
    num_points = len(x_s)
    return sum(x_s)//num_points, sum(y_s)//num_points

# https://stackoverflow.com/questions/15819980/calculate-mean-across-dimension-in-a-2d-array#15820027
def mean_2d(a):
    return map(lambda x:sum(x)/float(len(x)), zip(*a))

# https://stackoverflow.com/questions/30330945/applying-sobel-filter-on-image#30338007
def apply_sobel_filter(im):
    im = im.astype('int32')
    dx = nd.sobel(im,1)
    dy = nd.sobel(im,0)
    mag = np.hypot(dx,dy)
    mag *= 255.0/np.max(mag) 

    # fig, ax = plt.subplots()
    # ax.imshow(mag, cmap = 'gray')
    return mag


def interpolate_line_from_image(im, div_dim):
    """greyscale image input should already have some preprocessing filters applied"""
    kernel_dim = max(im.shape)//div_dim
    kernel = 1/(kernel_dim**2)*np.ones((kernel_dim, kernel_dim))
    result = strided_convolution_windows(im, kernel)
    threshold = np.mean(result)*.7
    result = binarized_image(result, threshold)
    return linear_regression_numpy(result)

def binarized_image(im, threshold):
    """greyscale image input"""
    img_np = np.array(im)
    img_np = np.where(img_np > threshold, 255, 0)
    im.putdata(img_np.flatten())
    return im

def linear_regression_numpy(binarized_image):
    """
    Fast linear regression using numpy's matrix operations.
    
    Parameters:
    points (np.ndarray): Binary array of shape (n, 2) containing x, y coordinates
    
    Returns:
    tuple: (slope, intercept)
    """
    points = [(x, y) for x, y in np.ndindex(binarized_image) if binarized_image[x, y]]
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

def strided_convolution_windows(image, kernel):
    """
    Alternative implementation using sliding_window_view for potentially better performance.
    
    Parameters:
    image_path (str): Path to the input image
    kernel (numpy.ndarray): 2D convolution kernel
    
    Returns:
    numpy.ndarray: Convolved image
    """
    # Load and convert image to grayscale numpy array
    img_array = np.array(image)
    
    # Get kernel dimensions
    kernel_height, kernel_width = kernel.shape
    
    # Create windows view
    windows = sliding_window_view(img_array, (kernel_height, kernel_width))
    
    # Extract windows with stride equal to kernel size
    windows = windows[::kernel_height, ::kernel_width]
    
    # Apply convolution to each window
    result = np.sum(windows * kernel, axis=(2, 3))
    
    return result


#edited from coursera robo
def points_local_to_world(theta, angles, ranges, ranger_position, world_pos):
    """
    theta-vehicle orientation in worldspace
    angles-local relative angles to entities
    ranges-local relative distance to entities
    ranger_position-camera position or origin on minimap
    world_pos-estimated world position of vehicle
    """
    xw, yw = world_pos
    #consider "ranger" to be either the camera or just the minimap center
    ranger_x, ranger_y, ranger_z = ranger_position
    # ranges = np.array(lidar.getRangeImage())
    # ranges = ranges[lidar_box:-lidar_box]  # slice off the invalid sides
    # ranges[ranges == np.inf] = 100

    w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                        [np.sin(theta),  np.cos(theta), yw],
                        [0, 0, 1]])
    lidar_samples = (len(angles), len(angles))
    
    X_i = np.array([ranges*np.cos(angles), ranges *
                    np.sin(angles), np.ones((lidar_samples,))])
    X_i = np.add(X_i, np.array([ranger_x, ranger_y, ranger_z]).reshape(3, 1))
    D = w_T_r @ X_i

# https://stackoverflow.com/questions/72900640/divide-an-image-into-radial-sections-and-select-values-in-an-individual-section
def radial_slices(orig, h, w, center, num_slices):
    cx, cy = center    # (x,y) coordinates of circle centre
    N      = num_slices         # number of slices in our pie
    l      = h + w       # length of radial lines - larger than necessary

    # orig = cv2.imread('artistic-swirl.jpg', cv2.IMREAD_ANYCOLOR)
    # print(orig.shape)

    # h, w   = 600, 1200   # image height and width
    # cx, cy = 200, 300    # (x,y) coordinates of circle centre
    # N      = 16          # number of slices in our pie
    # l      = h + w       # length of radial lines - larger than necessary

    rv = []
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
        # cv2.fillPoly(im, np.array([vertices],'int32'), 255)
        # cv2.imwrite(f'DEBUG-{sector:03d}.png', im)
        
        #cv2.imshow('title',im)
        #cv2.waitKey(0)
        
        mask = np.dstack((im,im,im))
        res = cv2.bitwise_and(mask,orig)
        # cv2.imshow('image',res)
        # cv2.waitKey(0)
        rv.append(res)
    return rv

# https://claude.ai/chat/30b0e915-39b7-4c49-81b4-d453b52c37b2
from scipy import signal

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

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# class ScreenRecorder(metaclass=Singleton):
#     def __init__(self):
#         self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
#         self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    # def  get_text(self, im):
    #     pixel_values = processor(images="image.jpeg", return_tensors="pt").pixel_values

    #     generated_ids = model.generate(pixel_values)
    #     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
import pytesseract
from PIL import ImageEnhance, ImageFilter
def text_from_image(im):
    im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('1')
    text = pytesseract.image_to_string(im)
    return text