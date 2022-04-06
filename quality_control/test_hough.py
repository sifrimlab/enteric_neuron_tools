import os
import numpy as np
from skimage import io
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from pyxelperfect.manipulate import automaticBrightnessAndContrast
from glob import glob

"""
    Idea is as follows: either i make the min distances very large to make sure the good images only get one peak, or allow for like 3 to 5 peaks and write
    some code to check if the two circles overlap
    --> update: 
"""
def plotStraightLine(edge_image, normal_image):
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    tested_angles = np.array([np.pi, np.pi / 2])
    h, theta, d = hough_line(edge_image, theta=tested_angles)

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(normal_image, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(edge_image, cmap="gray")
    ax[1].set_title('Input image')
    ax[1].set_axis_off()

    ax[2].imshow(edge_image, cmap="gray")
    ax[2].set_ylim((edge_image.shape[0], 0))
    ax[2].set_axis_off()
    accums, angles, dists = hough_line_peaks(h, theta, d)
    ax[2].set_title(f'{len(accums)} lines detected')
    for _, angle, dist in zip(accums, angles, dists):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

    plt.show()

def exactlyOneLine(edge_image):
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    tested_angles = [np.pi, np.pi / 2]
    h, theta, d = hough_line(edge_image, theta=tested_angles)
    accums, angles, dists =  hough_line_peaks(h, theta, d)
    # if not (len(accums) == len(angles) == len(dists) ):
    #     print(len(accums) , len(angles) , len(dists) )
    return len(accums) == 1

def exacltyOneCircle(edge_image):
    edge_image = canny(automaticBrightnessAndContrast(image)[0], sigma=3)
    hough_radii = np.arange(50, 75, 2)
    hough_res = hough_circle(edge_image, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_ydistance = 50, min_xdistance= 50 )
    return len(radii) == 1

def plotHoughCircles(edge_image):
    tmp_img = np.zeros((*edge_image.shape, 3))
    hough_radii = np.arange(50, 75, 2)
    hough_res = hough_circle(edge_image, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_ydistance = 50, min_xdistance= 50 )

    fig, axs = plt.subplots(1,3)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,shape=tmp_img.shape)
        tmp_img[circy, circx] = (220, 20, 20)
    axs[0].imshow(image)
    axs[1].imshow(edge_image)
    axs[2].imshow(tmp_img)
    axs[2].set_title(f"{len(radii)}")
    plt.show()

# for i, img in enumerate(glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary/*.tif")):
#     image = io.imread(img)
    # edge_image = canny(image, sigma=0)
    # # if exacltyOneCircle(edge_image):
    # if not exactlyOneLine(edge_image):
    #     io.imsave(f"./good/{os.path.basename(img)}", image)
    # else:
    #     io.imsave(f"./bad/{os.path.basename(img)}", image)
    # if i > 200:
    #     break

for i, img in enumerate(glob("./bad/*")): 
# for i, img in enumerate(glob("./good/*")): 
    image = io.imread(img)
    edge_image = canny(image, sigma=0)
    plotStraightLine(edge_image, image)
    # plotHoughCircles(edge_image)
    if i > 15:
        break
