import os
import numpy as np
from skimage import io
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from pyxelperfect.manipulate import automaticBrightnessAndContrast
from glob import glob

"""
    Idea is as follows: either i make the min distances very large to make sure the good images only get one peak, or allow for like 3 to 5 peaks and write
    some code to check if the two circles overlap
"""

# Load picture and detect edges
# two_nuclei = img_as_ubyte(io.imread("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary/bad/aligned_tile457_cell111.tif"))
# cut_corner = img_as_ubyte(io.imread("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary/bad/aligned_tile186_cell122.tif"))
# nothing = img_as_ubyte(io.imread("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary/bad/aligned_tile18_cell99.tif"))
# good_image =  img_as_ubyte(io.imread("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary/good/aligned_tile70_cell47.tif"))

# edges = canny(automaticBrightnessAndContrast(image)[0], sigma=5)
# two_nuclei_edges = canny(automaticBrightnessAndContrast(two_nuclei)[0], sigma=5)
# cut_corner_edges = canny(automaticBrightnessAndContrast(cut_corner)[0], sigma=5)
# nothing_edges     = canny(nothing, sigma=5)
# good_image_edges = canny(automaticBrightnessAndContrast(good_image)[0], sigma=5)

def exacltyOneCircle(image):
    edge_image = canny(automaticBrightnessAndContrast(image)[0], sigma=5)
    hough_radii = np.arange(20, 65, 2)
    hough_res = hough_circle(edge_image, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_ydistance = 50, min_xdistance= 50 )
    return len(radii) == 1


for i, img in enumerate(glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary/*.tif")):
    image = io.imread(img)
    if exacltyOneCircle(image):
        io.imsave(f"./good/{os.path.basename(img)}", image)
    else:
        io.imsave(f"./bad/{os.path.basename(img)}", image)
    
    if i > 20:
        break



# Detect two radii
# for i, img in enumerate((two_nuclei_edges, cut_corner_edges, nothing_edges, good_image_edges)):
#     hough_radii = np.arange(20, 65, 2)
#     tmp_img = np.zeros((*img.shape, 3))
#     hough_res = hough_circle(img, hough_radii)
#     # io.imsave(f"{i}.tif", hough_res)

# # Select the most prominent 3 circles
#     accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_ydistance = 50, min_xdistance= 50 )

# # Draw them
#     fig, ax = plt.subplots(ncols=1, nrows=1)
#     # img = color.gray2rgb(img)
#     for center_y, center_x, radius in zip(cy, cx, radii):
#         circy, circx = circle_perimeter(center_y, center_x, radius,shape=img.shape)
#         tmp_img[circy, circx] = (220, 20, 20)
#         # img[circy, circx] = 220

#     ax.imshow(tmp_img, cmap=plt.cm.gray)
#     # plt.savefig(f"{i}_hough.tif")
#     plt.show()

