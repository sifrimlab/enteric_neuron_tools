import os
import sys
import time
import argparse
import numpy as np
import numpy as np
import aicspylibczi
from glob import glob
from tqdm import tqdm
from skimage import io
from tiling import tile
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import laplace
import custom_io, analysis, processing

# Input parameters
file_path = "./2024_07_17__16_07__0015_M4 C.czi"
out_dir = Path("./out_dir")
out_dir.mkdir(exist_ok=True)

z_number = 0
c_number = 2
pixel_density = 3.2
sigma = 7
min_samples = 2

tile_size = 2048

# End input, start tiling of large image

filename_base = Path(file_path).stem

czi = aicspylibczi.CziFile(file_path)

img_extracted = czi.read_mosaic(C=c_number, Z=z_number)[0, 0, :, :]

io.imsave(out_dir / f"{filename_base}_c{c_number}_z{z_number}.tif", img_extracted)

tile(str(out_dir / f"{filename_base}_c2_z0.tif"), tile_size, tile_size, out_dir = out_dir)

# Now that we have the tiles, we're gonna count the cells

# Quick and dirty function to create the neurons, meta and directory variables used by the rest of the COUNTEN script
def readCOUNTENimage(filename: str, neurons: np.array):
    # neurons = io.imread(filename)
    meta = {"Name": os.path.basename(os.path.splitext(filename)[0])}
    directory = os.path.join(str(out_dir), f"result_{os.path.splitext(meta['Name'])[0]}")
    return neurons, meta, directory

# Quick and dirty function to do the processing once neurons, meta and directoyr exist
def processCOUNTENdata(neurons, meta, directory):
    if os.path.exists(directory):
        expand = 0
        while True:
            expand += 1
            new_directory = directory+"_"+str(expand)
            if os.path.exists(new_directory):
                continue
            else:
                directory = new_directory
                os.makedirs(directory, exist_ok=True)
                break
    else:
        os.makedirs(directory, exist_ok=True)

    # Actually process the image and segmetn
    local_maxi, labels, gauss = processing.wide_clusters(neurons,
                                                         sigma=sigma,
                                                         pixel_density=pixel_density,
                                                         min_samples=min_samples,
                                                         meta=meta,
                                                         directory=directory,
                                                         save= True)

    ganglion_prop = processing.segmentation(gauss, local_maxi, labels, meta, directory, save = True)

    # Save the dataframe
    # Run dataframe function from module
    _, _ = analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)



for tile_img_path in tqdm(glob(str(out_dir / "*tile*.tif"))):
    neurons, meta, directory = readCOUNTENimage(tile_img_path, io.imread(tile_img_path))
    processCOUNTENdata(neurons,meta,directory)
    plt.close()
