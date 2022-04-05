import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from pyxelperfect.manipulate import automaticBrightnessAndContrast


# Load picture and detect edges
image = img_as_ubyte(io.imread("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary/bad/aligned_tile457_cell111.tif"))
edges = canny(automaticBrightnessAndContrast(image)[0], sigma=5)

# plt.imshow(edges)
# plt.show()


# Detect two radii
hough_radii = np.arange(20, 65, 2)
hough_res = hough_circle(edges, hough_radii)

# plt.imshow(hough_res[-1, :, :])
# plt.show()
# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=3)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()

