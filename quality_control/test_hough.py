import os
import numpy as np
import random
from tqdm import tqdm
from shutil import copyfile
from skimage import io
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.filters import sobel
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
# from pyxelperfect.manipulate import automaticBrightnessAndContrast
from glob import glob
from joblib import Parallel, delayed

def plotStraightLine(edge_image, normal_image):
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
    tested_angles = np.array([np.pi, np.pi / 2])
    h, theta, d = hough_line(edge_image, theta=tested_angles)
    accums, angles, dists =  hough_line_peaks(h, theta, d)
    return len(accums) == 1

def exacltyOneCircle(edge_image):
    """
    For this, it is VERY important that the input img is 16bit, and that the edge filter used is sobel
    """
    hough_radii = np.arange(25, 75, 2)
    hough_res = hough_circle(edge_image, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_ydistance = 50, min_xdistance= 50 )
    return len(radii) == 1

def plotHoughCircles(edge_image):
    tmp_img = np.zeros((*edge_image.shape, 3))
    hough_radii = np.arange(25, 75, 2)
    hough_res = hough_circle(edge_image, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_ydistance = 50, min_xdistance= 50 )

    fig, axs = plt.subplots(1,3)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,shape=tmp_img.shape)
        tmp_img[circy, circx] = (220, 20, 20)
    axs[0].imshow(image)
    axs[1].imshow(edge_image)
    # axs[1].set_title(f"low = {low}, high = {high}")
    axs[2].imshow(tmp_img)
    axs[2].set_title(f"{len(radii)}")
    plt.show()

already_done_basename_list = glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/SEP/datasets/colon/single_nuclei/line_qcd/*d/*.tif")
def fix_stuff(img):
    if os.path.basename(img) in already_done_basename_list:
        return None
    try:
        image = io.imread(img)
    except: 
        return None
    sobel_image = sobel(image)
    sobel_image = img_as_ubyte(sobel_image)
    # if exacltyOneCircle(sobel_image):
    if not exactlyOneLine(sobel_image):
        copyfile(img,f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/SEP/datasets/colon/single_nuclei/line_qcd/good/{os.path.basename(img)}" )
        return "good"
        # io.imsave(f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary_line_qcd/good/{os.path.basename(img)}", image)
    else:
        copyfile(img,f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/SEP/datasets/colon/single_nuclei/line_qcd/bad/{os.path.basename(img)}" )
        return "bad"
            # io.imsave(f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary_line_qcd/bad/{os.path.basename(img)}", image)

# if __name__ == '__main__':
#     file_list = glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/SEP/datasets/colon/single_nuclei/*")
#     for file in subset_list:
#         if file.endswith(".tif"):
#             fix_stuff(file)
Parallel(n_jobs=16)(delayed(fix_stuff)(img) for img in tqdm(glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/SEP/datasets/colon/single_nuclei/*.tif")))
