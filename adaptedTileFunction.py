import os
import glob
import numpy as np
from skimage import io
from typing import List
from tifffile import imsave
from skimage.io import imread_collection, imsave, imread

def calculateOptimalLargestResolution(images: str, target_tile_width: int, target_tile_height: int) -> List[int]: 
    images_array = np.array(io.ImageCollection(images))

    heights = []
    widths = []
    for i in range(len(images_array)):
        heights.append(images_array[i].shape[0])
        widths.append(images_array[i].shape[1])
    max_rows = max(heights) 
    max_columns = max(widths)

    ydiv = np.ceil(max_rows / target_tile_height)
    xdiv = np.ceil(max_columns / target_tile_width)

    target_full_rows = ydiv * target_tile_height
    target_full_columns = xdiv * target_tile_width

    return int(target_full_rows), int(target_full_columns), int(ydiv), int(xdiv)

def padImage(image_path: str, target_full_rows: int, target_full_columns: int) -> np.ndarray:
    image = imread(image_path)

    rowdiff = target_full_rows - image.shape[0]
    columndiff = target_full_columns - image.shape[1]
    padded_img = np.pad(image, ((0, rowdiff), (0, columndiff)))
    return padded_img

def tileImage(image, ydiv: int, xdiv: int, image_prefix: str="test"):
    temp_split = np.array_split(image, ydiv, axis = 0)
    # Item sublist part is just to unpack a list of lists into one list
    final_split = [item for sublist in [np.array_split(row, xdiv, axis = 1) for row in temp_split] for item in sublist]

    for i, img in enumerate(final_split, 1):
        imsave(f"{image_prefix}_tile{i}.tif", img) 
    return len(final_split)

def globalTilingFunc(glob_pattern, target_tile_width:str, target_tile_height:str, image_prefix: str):
    target_full_rows, target_full_columns, ydiv, xdiv = calculateOptimalLargestResolution(glob_pattern, target_tile_width, target_tile_height)
    padded_imgs = {}
    for image_path in glob.glob(glob_pattern):
        padded_imgs[image_prefix] = padImage(image_path, int(target_full_rows), int(target_full_columns))

    for k, padded_img in padded_imgs.items():
        n_tiles = tileImage(padded_img, ydiv = ydiv, xdiv = xdiv, image_prefix = k)
    return n_tiles # return number of tiles to be sued by processing iterator
        
