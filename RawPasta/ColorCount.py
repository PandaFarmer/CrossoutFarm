# https://stackoverflow.com/questions/69932895/count-total-number-of-pixels-for-each-color
#!/usr/bin/env python3
from PIL import Image
import numpy as np

# Open image and ensure RGB
im = Image.open('UMN9c.png').convert('RGB')

# Make into Numpy array
na = np.array(im)

# Get colours and corresponding counts
colours, counts = np.unique(na.reshape(-1,3), axis=0, return_counts=1)

print(colours, counts)