import sys
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import custom_io, analysis, processing
from skimage import io
from tiling import completeTiling

# First extract the correct image from the tif stack

ap = argparse.ArgumentParser(description="Extract tiff from a .tif image and count neurons and ganglia.")
ap.add_argument('tif_path',type=str,help="Path (relative or absolute) to target image")
ap.add_argument('-o', '--out_dir', type=str, help="Root directory where output should be stored, default is base dir of the input image")
# ap.add_argument('-c', '--c_number', default=0, type=int, help="indexes (start at 0) of the channel contains the marker of interest. Default = 0.")

# Parameters

ap.add_argument('-p', '--pixel_density', default=3.2, type=float, help="Pixel density of the image (pixels/micrometer. default = 3.2")
ap.add_argument('-s', '--sigma', default=7, type=int, help="Sigma used to smooth the image using a gaussian smoother. default = 7")
ap.add_argument('-m', '--min_samples', default=2, type=int, help="Minimum number of neurons in a ganglion. default = 2")

# ap.add_argument('-i', '--maxIP', default=True,action="store_false", help="Flag that turns off taking the maxIP of the z-dimension. If used, requires the usage of --z_number")
# ap.add_argument('-z', '--z_number', default=None, type=int, help="index (start at 0) of the z-stack that needs to be extracted.")

args = ap.parse_args()

# if no out_dir is given, take the base dir of the input image
if args.out_dir is None:
    args.out_dir = os.path.dirname(args.tif_path)


filename = args.tif_path
filename_base = os.path.splitext(os.path.basename(filename))[0]

# make output_dir if it doesn't exist
os.makedirs(args.out_dir, exist_ok=True)


### Analyzing

neurons =  io.imread(filename)
meta = {"Name": os.path.splitext(filename_base)[0]}
directory = os.path.join(str(args.out_dir), f"result_{os.path.splitext(meta['Name'])[0]}_{time.strftime('%m'+'_'+'%d'+'_'+'%Y')}")


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

if any(dim > 2000 for dim in neurons.shape):
    tiling_name_list = completeTiling(neurons, filename_base)
else:
    tiling_name_list = [filename]

for tile_name in tiling_name_list:
    neurons = io.imread(tile_name)
    meta = {"Name": os.path.splitext(tile_name)[0]}
    # We don't redefine directory cause we want it to end up in the same directory
    # Actually process the image and segmetn
    local_maxi, labels, gauss = processing.wide_clusters(neurons,
                                                         sigma=args.sigma,
                                                         pixel_density=args.pixel_density,
                                                         min_samples=args.min_samples,
                                                         meta=meta,
                                                         directory=directory,
                                                         save= True)

    ganglion_prop = processing.segmentation(gauss, local_maxi, labels, meta, directory, save = True)

    # Save the dataframe

    # Run dataframe function from module
    _, _ = analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)
