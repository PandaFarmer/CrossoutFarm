# https://stackoverflow.com/questions/30330945/applying-sobel-filter-on-image#30338007

import scipy.ndimage as nd
import cv2
import numpy as np
import matplotlib.pyplot as plt

#code to apply sobel using scipy
im = nd.imread('ygqCd.jpg', True)
im = im.astype('int32')
dx = nd.sobel(im,1)
dy = nd.sobel(im,0)
mag = np.hypot(dx,dy)
mag *= 255.0/np.max(mag) 

fig, ax = plt.subplots()
ax.imshow(mag, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.show()

#code to apply multiple filter variations using cv2..

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

img = cv2.imread('ygqCd.jpg', 0)
blurred_img = cv2.medianBlur(img, 11)

s_mask = 17

sobelx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=s_mask))
b_sobelx = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=s_mask))
sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
b_sobelx = interval_mapping(b_sobelx, np.min(sobelx), np.max(sobelx), 0, 255)

sobely = np.abs(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=s_mask))
sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
b_sobely = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=s_mask))
b_sobely = interval_mapping(b_sobely, np.min(sobely), np.max(sobely), 0, 255)

sobel_xy = 0.5 * sobelx + 0.5 * sobely
b_sobel_xy = 0.5 * b_sobelx + 0.5 * b_sobely

fig = plt.figure(figsize=(10, 14))
plt.subplot(3,2,1),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(b_sobelx,cmap = 'gray')
plt.title('Blurred Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(b_sobely,cmap = 'gray')
plt.title('Blurred Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(sobel_xy,cmap = 'gray')
plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(b_sobel_xy,cmap = 'gray')
plt.title('Blurred Sobel XY'), plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()