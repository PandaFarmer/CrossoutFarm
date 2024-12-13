import scipy
from scipy.ndimage import gaussian_filter
from PIL import Image
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

image_filenames= ["road_hilly.jpg",
"road_left_far.jpg",
"road_left_offset.jpg",
"road_left_offset_mild.jpg",
"road_right_far.jpg",
"road_right_offset.jpg",
"road_s.jpg",
"road_straight.jpg"]
# for image_filename in image_filenames:
image_filename = "road_hilly.jpg"
im = Image.open("Roads/" + image_filename)
im = im.convert('L')#greyscale
#cv2 behaving strangely, too lazy to fix
sobel_kernel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
sobel_kernel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

def interval_mapping(image, from_min, from_max, to_min=0, to_max=255):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

b_im =  gaussian_filter(im, sigma = 5)

G_x = scipy.signal.convolve2d(im, sobel_kernel_x)
G_x = interval_mapping(G_x, np.min(G_x), np.max(G_x))
G_y = scipy.signal.convolve2d(im, sobel_kernel_y)
G_y = interval_mapping(G_y, np.min(G_y), np.max(G_y))

b_G_x = scipy.signal.convolve2d(b_im, sobel_kernel_x)
b_G_x = interval_mapping(b_G_x, np.min(G_x), np.max(G_x))
b_G_y = scipy.signal.convolve2d(b_im, sobel_kernel_y)
b_G_y = interval_mapping(b_G_y, np.min(G_x), np.max(G_x))

# sqrt_all = np.vectorize(lambda v: sqrt(v))
# G = sqrt_all(b_G_x*b_G_x + b_G_y*b_G_y)

G_xy = 0.5 * G_x + 0.5 * G_y
b_G_xy = 0.5 * b_G_x + 0.5 * b_G_y

scale_factor = 255
offset = 0
G_x = scale_factor*G_x + offset
G_y = scale_factor*G_y + offset
b_G_x = scale_factor*b_G_x + offset
b_G_y = scale_factor*b_G_y + offset
G_xy = scale_factor*G_xy + offset
b_G_xy = scale_factor*b_G_xy + offset

plt.subplot(3,2,1),plt.imshow(G_x,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(b_G_x,cmap = 'gray')
plt.title('Blurred Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(G_y,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(b_G_y,cmap = 'gray')
plt.title('Blurred Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(G_xy,cmap = 'gray')
plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(b_G_xy,cmap = 'gray')
plt.title('Blurred Sobel XY'), plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()