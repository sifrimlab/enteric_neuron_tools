import os
from glob import glob
import re
import numpy as np
import pandas as pd
from skimage import io
from tifffile import imsave
from typing import List, Tuple
import matplotlib.pyplot as plt
import imagesize
from skimage.io import imread_collection, imsave, imread


def calculateOptimalLargestResolution(glob_pattern: str, target_tile_height: int, target_tile_width: int) -> Tuple[int]: 
    """Calculates the optimal maximum resolution based on a set of images assuming that it will be used to tile the images into equal tiles of a certain size, given by the input variables.


    Parameters
    ----------
    glob_pattern : str
        glob pattern hitting all images to be taken into account for the calculation.
    target_tile_height : int
        target_tile_height
    target_tile_width : int
        target_tile_width

    Returns
    -------
    Tuple[int]

    """
    
    def calcMaxesOfNonLoadedImages(str_glob_pattern):
        heights = []
        widths = []
        for file in glob(str_glob_pattern):
            width, height = imagesize.get(file)
            widths.append(width)
            heights.append(height)
        return max(heights), max(widths)

    def calcMaxes(images_array):
        heights = []
        widths = []
        if len( images_array.shape ) > 2:
            for i in range(len(images_array)):
                heights.append(images_array[i].shape[0])
                widths.append(images_array[i].shape[1])
            max_rows = max(heights) 
            max_columns = max(widths)
        else:
            max_rows=images_array.shape[0]
            max_columns=images_array.shape[1]
        return max_rows, max_columns

    if isinstance(glob_pattern, str):
        # images_array = np.array(imread_collection(glob_pattern))
        max_rows, max_columns = calcMaxesOfNonLoadedImages(glob_pattern)
    elif isinstance(glob_pattern, np.ndarray):
        images_array = glob_pattern
        max_rows, max_columns = calcMaxes(images_array)
    elif isinstance(glob_pattern, tuple):
        # If the given argument is just already the max resolution
        max_rows, max_columns = glob_pattern

    rowdiv = np.ceil(max_rows / target_tile_height)
    coldiv = np.ceil(max_columns / target_tile_width)

    target_full_rows = rowdiv * target_tile_height
    target_full_columns = coldiv * target_tile_width

    return target_full_rows, target_full_columns, rowdiv, coldiv

def padImage(image: np.array, target_full_rows: int, target_full_columns: int) -> np.ndarray:
    """Pads an image with black pixels up until a certain number of rows and columns

    Parameters
    ----------
    image : np.array
        image to be padded
    target_full_rows : int
        Number of rows to be achieved by padding
    target_full_columns : int
        Number of columns to be achieved by padding

    Returns
    -------
    np.ndarray

    """
    rowdiff = target_full_rows - image.shape[0]
    columndiff = target_full_columns - image.shape[1]
    padded_img = np.pad(image, ((0, rowdiff), (0, columndiff)))
    return padded_img

def tileImage(image: np.ndarray, rowdiv: int, coldiv: int, image_prefix: str="image_tile_"):
    """Tile an image into smaller tiles based on divisions in x and y axes, and saving them to tif files.

    Parameters
    ----------
    image : np.ndarray
        image
    rowdiv : int
        Nr of divisions to make in the y axis
    coldiv : int
        Nr of divisions to make in the x axis
    image_prefix : str
        prefix to add to the new filenames of the tiles. default = image_tile
    """
    if not image_prefix.endswith("_"):
        image_prefix += "_"

    temp_split = np.array_split(image, rowdiv, axis = 0)
    # Item sublist part is just to unpack a list of lists into one list
    final_split = [item for sublist in [np.array_split(row, coldiv, axis = 1) for row in temp_split] for item in sublist]

    for i, img in enumerate(final_split, 1):
        imsave(f"{image_prefix}tile{i}.tif", img, check_contrast=False)

def tile(glob_pattern: str, target_tile_width: int, target_tile_heigt: int, out_dir: str = "", calc_only=False) -> Tuple[int]:
    """Tile the images caught by the glob pattern by first padding them to a global image size needed to tile them all into the same tile size, given by the input values. Tiles are written to tif files in the basedir of the original image, or to out_dir if given, with naming convention {out_dir | basedir}/{image_name}_tile{i}.tif

    Parameters
    ----------
    glob_pattern : str
        glob_pattern that catches all images to be tiled.
    target_tile_width : int
        Width of the tiles
    target_tile_height : int
        Height of the tiles
    calc_only: bool
        If true, only calculates the padded image size and number of tiles in both dimensions, without creating the tiles. 
        Output can be used to create a tileGrid object. 
        

    Returns
    -------
    Tuple[int]

    """
    if out_dir and not calc_only:
        os.makedirs(out_dir, exist_ok=True)

    target_full_rows, target_full_columns, rowdiv, coldiv = calculateOptimalLargestResolution(glob_pattern, target_tile_height, target_tile_width)

    if calc_only:
        grid = tileGrid(rowdiv, coldiv, target_full_rows, target_full_columns)
        return grid

    padded_imgs = {}
    if not out_dir:
        for image_path in glob(glob_pattern):
            padded_imgs[os.path.splitext(image_path)[0]] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))
    else:
        for image_path in glob(glob_pattern):
            padded_imgs[os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0])] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))

    for k, padded_img in padded_imgs.items():
        tileImage(padded_img,rowdiv = rowdiv, coldiv = coldiv, image_prefix = k)
    grid = tileGrid(rowdiv, coldiv, target_full_rows, target_full_columns)
    return rowdiv,coldiv, target_full_rows, target_full_columns

class tileGrid:
    # """ Class to represent the tiling grid of the tile function
    # Important for usage: tile names start at indexin 1, so for self.tile_boundaries (which is a dict), keys start at 1
    # image list and data coordinates are there to be able to tile them after the fact
    # """
    def __init__(self, rowdiv, coldiv, n_rows, n_cols, image_list = [], data_coordinates_list=[], tiles={}, pattern="row_by_row"):
# basic vars
        self.n_rows = n_rows # Number rows in the complete array, untiled
        self.n_cols = n_cols # Number cols in the complete array, untiled
        self.coldiv = int( coldiv ) # Number of tiles in the col-dimension
        self.rowdiv = int( rowdiv ) # Number of tiles in the row dimension
        self.image_list = image_list # List of images of the original image-stack ,in case the tiling tiles more than 1 image in unison
        self.data_coordinates_list = data_coordinates_list # List of potential dataframes with point coordinates, such as spatial transcriptomics
        self.tiles = tiles

        # calculated stuff
        self.tile_size_row =int(  n_rows / rowdiv ) # Size of each individual tile in row dimension
        self.tile_size_col =int(  n_cols / coldiv ) # Size of each individual tile in col dimension
        self.n_tiles = int(rowdiv * coldiv) # total number of tiles created

        # numerical representation of the tiles
        if pattern == "row_by_row":
            self.tile_grid = np.arange(1, self.n_tiles + 1).reshape(self.rowdiv, self.coldiv) # representation of the tile grid in numbers, for if you want to search for the position of a specific tile
        elif pattern == "snake_by_row":
            '''
            [[ 1 2 3 ]
             [ 6 5 4 ]
             [ 7 8 9 ]]
            '''
            arr = np.arange(1, self.n_tiles+1)
            # Reshape the range into a 2D array
            arr_2d = arr.reshape(self.rowdiv, self.coldiv)
            # Reverse every other row
            for i in range(1, arr_2d.shape[0], 2):
                arr_2d[i] = arr_2d[i, ::-1]
            # Print the resulting array
            self.tile_grid = arr_2d

        # Calculate boundaries of tiles
        self.tile_boundaries = {} # Dict that stores the boundaries of the tiles (with respect to the padded image, not the original)
        # start at one to make the boundary math check out
        for i in range(1, self.n_tiles + 1):
            idx = np.where(self.tile_grid == i)
            self.tile_boundaries[i] = np.s_[idx[0][0] * self.tile_size_row : (idx[0][0] + 1) * self.tile_size_row, idx[1][0] * self.tile_size_col: (idx[1][0] + 1) * self.tile_size_col]

    def addImage(self, image: np.ndarray):
        self.image_list.append(image)

    def addTile(self, tile: "Tile"):
        self.tiles[tile.tile_nr] = tile

    def getTile(self, tile_nr):
        return self.tiles[tile_nr]

    def plotImageTile(self, tile_nr = 1, image_nr = 0):
        plt.imshow(self.image_list[image_nr][self.tile_boundaries[tile_nr]])
        plt.show()

    def getImageTile(self, tile_nr = 1, image_nr = 0):
        return self.image_list[image_nr][self.tile_boundaries[tile_nr]]

    def addDataCoordinates(self, data_df):
        self.data_coordinates_list.append(data_df)

    def getTileDataCoordinates(self, tile_nr, data_index=0,rowname="row", colname="col"):
        """getTileDataCoordinates.
        Note to self: because padding happens only at the end of the dimensions, it's not included in the tiling of coordinates, since their coordinate is the same relative to the padded image.

        Parameters
        ----------
        tile_nr :
            nr of tile to fetch coordinates from. (indexing starts at 1, not at 0)
        data_index :
            index of which dataframe to fetch, in case multiple dataframes are linked with the tiling coordinate. (indexing starts at 0)
        rowname :
            column name that refers to the row dimension
        colname :
            column name that refers to the col dimension
        """

        df = self.data_coordinates_list[data_index]
        cropped_df = self._cropCoordinateDf(df, tile_nr, rowname=rowname, colname=colname)

        return cropped_df

    def _cropCoordinateDf(self, df, tile_nr, rowname="row", colname="col"):
        row_slice, col_slice =  self.tile_boundaries[tile_nr][0], self.tile_boundaries[tile_nr][1]

        tmp_df = df.loc[(df[rowname] >= row_slice.start) & (df[rowname] < row_slice.stop)]
        cropped_df = tmp_df.loc[(tmp_df[colname] >= col_slice.start) & (tmp_df[colname] < col_slice.stop)]

        local_rows = [el - row_slice.start for el in cropped_df[rowname]]
        local_cols = [el - col_slice.start for el in cropped_df[colname]]

        # copy line added since assigning new columns to the normal cropped_df (which is technically a slice of the original df) raises a warning.
# In this usecase the warning is a false positive, since we don't care about tracing the new column back to the original dataframe
        cropped_df = cropped_df.copy()
        cropped_df["local_row"] = local_rows
        cropped_df["local_col"] = local_cols
        return cropped_df 

    def __str__(self):
        return f"Tile grid of size {self.rowdiv} by {self.coldiv}, {self.n_tiles} in total.\nTiles are {self.tile_size_row} rows by {self.tile_size_col} cols.\n {self.tile_grid}"

    def getNeighbouringTiles(self, tile_nr):
        flattened_grid = self.tile_grid.flatten()
        idx = np.argwhere(self.tile_grid == tile_nr)[0]
        left = tile_nr - 1 if (tile_nr - 1  in flattened_grid) and (idx[1] !=  0) else None
        right =  tile_nr + 1 if (tile_nr + 1 in flattened_grid) and (idx[1] != self.coldiv - 1) else None
        top =  tile_nr - self.coldiv if (tile_nr - self.coldiv in flattened_grid) and (idx[0] != 0) else None
        bot =  tile_nr + self.coldiv if (tile_nr + self.coldiv in flattened_grid) and (idx[0] != self.rowdiv-1) else None
        return {"left": left, "top": top, "bot": bot, "right": right}

