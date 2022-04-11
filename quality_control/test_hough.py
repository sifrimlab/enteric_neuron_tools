import os
import numpy as np
import itertools
import random
from tqdm import tqdm
from shutil import copyfile
from skimage import io
from skimage import data, color
from skimage.transform import hough_circle, hough_ellipse, hough_circle_peaks, hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.filters import sobel, gaussian
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte, img_as_bool
import matplotlib.pyplot as plt
from pyxelperfect.manipulate import automaticBrightnessAndContrast
from glob import glob
from joblib import Parallel, delayed

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

def plotHoughEllipses(edge_image):
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edge_image, accuracy=20, threshold=250,
                           min_size=100, max_size=120)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, axs = plt.subplots(ncols=3, nrows=1, sharex=True,
                                    sharey=True)

    axs[0].set_title('Original picture')
    axs[0].imshow(image)

    axs[1].set_title('edge image')
    axs[1].imshow(edge_image)

    axs[2].set_title(f"{len(result[0])}")
    axs[2].imshow(edges)

    plt.show()
# already_done_basename_list =[os.path.basename(file) for file in glob(f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/LiverSample_line_qcd/*/*.tif")]
# def fix_stuff(img):
    # if os.path.basename(img) in already_done_basename_list:
    #     return None
    # image = io.imread(img)
    # edge_image = canny(image, sigma=0)
    # if exacltyOneCircle(edge_image):
    # if not exactlyOneLine(edge_image):
        # copyfile(img,f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/LiverSample_line_qcd/good/{os.path.basename(img)}" )
        # return "good"
        # io.imsave(f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary_line_qcd/good/{os.path.basename(img)}", image)
    # else:
        # copyfile(img,f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/LiverSample_line_qcd/bad/{os.path.basename(img)}" )
        # return "bad"
            # io.imsave(f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/brainimagelibrary_line_qcd/bad/{os.path.basename(img)}", image)

# results = Parallel(n_jobs=16)(delayed(fix_stuff)(img) for img in tqdm(glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/LiverSample/*.tif")))

# goodish_imgs = glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/LiverSample_line_qcd/good/*")
# random.seed(12)
# random.shuffle(goodish_imgs)


# for i, img in enumerate(goodish_imgs): 
#     image = io.imread(img)
    # sobel_image = sobel(image)
    # sobel_image = img_as_ubyte(sobel_image)
    # sobel_image = gaussian(sobel_image, sigma=3)

    # plotHoughCircles(sobel_image)
    # plt.show()
    # if i > 5:
    #     break


# output_list = list(itertools.product(range(0,255,1), range(0,255,1)))
# output_list = [(el[0], el[1]) for el in output_list if el[0] < el[1] ]

# # highs =list(np.arange(1,255, 1))
# # doubled_highs = highs.copy()

# # new_low_value = 0
# # new_high_value = 1
# # lows = [new_low_value for i in range(len(highs))]
# # for i in range(len(highs)):
# #     lows.append(new_low_value)
# #     if 
# #     new_low_value += 1
# #     doubled_highs.append(new_high_value)
# #     if new_low_value >= new_high_value:
# #         new_high_value += 1
    

# slideshow = np.zeros((149,149,len(output_list)))

# for i, (low, high) in tqdm(enumerate(output_list)):
#     print(low, high)
#     edge_image = canny(image, low_threshold=low, high_threshold=high, sigma=0)
#     slideshow[:,:, i] = edge_image
# # # plotStraightLine(edge_image, image)
# #     # plotHoughCircles(edge_image)
# io.imsave("slideshow.tif", slideshow)
