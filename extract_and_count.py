import sys
import os
import time
import argparse
import numpy as np
import importlib.util # This is to manually import the scripts to make sure it's all correct
import matplotlib.pyplot as plt
import numpy as np
import aicspylibczi
import custom_io, analysis, processing
from skimage import io
from skimage.filters import laplace
from adaptedTileFunction import globalTilingFunc
from icecream import ic


# First extract the correct image from the czi stack

ap = argparse.ArgumentParser(description="Extract tif from an image and count neurons and ganglia.")
ap.add_argument('file_path',type=str,help="Path (relative or absolute) to target image")
ap.add_argument('-o', '--out_dir', type=str, help="Root directory where output should be stored, default is base dir of the input image")
ap.add_argument('-c', '--c_number', default=0, type=int, help="indexes (start at 0) of the channel contains the marker of interest. Default = 0.")

# Parameters

ap.add_argument('-p', '--pixel_density', default=3.2, type=float, help="Pixel density of the image (pixels/micrometer. default = 3.2")
ap.add_argument('-s', '--sigma', default=7, type=int, help="Sigma used to smooth the image using a gaussian smoother. default = 7")
ap.add_argument('-m', '--min_samples', default=2, type=int, help="Minimum number of neurons in a ganglion. default = 2")

ap.add_argument('-i', '--maxIP', default=False,action="store_true", help="Flag that turns on taking the maxIP of the z-dimension.")
ap.add_argument('-z', '--z_number', default=None, type=int, help="index (start at 0) of the z-stack that needs to be extracted.")
ap.add_argument('-t', '--tile_size', default=None, type=int, nargs=2, help="Tuple representing targetted tile size (X-Y). Example: `-t 2000 2000`. Default is no tiling behaviour")

args = ap.parse_args()
# if no out_dir is given, take the base dir of the input image
if args.out_dir is None:
    args.out_dir = os.path.dirname(args.file_path)

# make output_dir if it doesn't exist
os.makedirs(args.out_dir, exist_ok=True)

if args.z_number and args.maxIP:
    ap.error('You cannot both take a maxIP and extract a single z-stack')

extract_most_in_focus = bool(not args.maxIP and args.z_number is None) # if neither flags are set, we just extract the most in focus z-stack

def getMostInFocusImage(image_array_list):
    stdev_list = []
    for image in image_array_list:
        # calculate edges in image
        edged = laplace(image)
        # Calculate stdev of edges
        stdev = np.std(edged)
        stdev_list.append(stdev)
    
    # Find largest stdev in list
    largest = max(stdev_list)
    # Fidn which index it is to link back to the original list
    index = stdev_list.index(largest)
    print("Extracted most in focus z-stack is index {index}")
    return image_array_list[index], index



filename = args.file_path
if filename.lower().endswith("czi"):
    image_type = "czi"
elif filename.lower().endswith(("tif", "tiff")):
    image_type = "tif"

filename_base = os.path.splitext(os.path.basename(filename))[0]


def maxIPstack(img_list):
    parsed_list = img_list
    parsed_list = [img if isinstance(img, np.ndarray) else io.imread(img) for img in img_list]
    # now all elements in parsed_list are ndarrays
    maxIP = np.maximum.reduce(parsed_list)
    return maxIP

# Now we extract the image in question, workflow of this has to be different depending on the filetype
if image_type == "czi":

    czi = aicspylibczi.CziFile(filename)

    z_min, z_max = czi.get_dims_shape()[0]['Z']
    z_numbers = range(z_min,z_max)

    image_list = []
    for z_num in z_numbers:
        # Extract all z-stacks and put them in list for maxIP function
        image_slice, shape = czi.read_image(C=args.c_number, Z=z_num)
        image_slice = image_slice[0,0,0,0,0,:,:]
        image_list.append(image_slice)

    # If the user wants a maxip
    if args.maxIP:
        img_extracted = maxIPstack(image_list)

    # If the user doesn't want a maxIP, we extract the most in focus z-stack
    elif extract_most_in_focus:
        img_extracted, args.z_number =  getMostInFocusImage(image_list)
    # if 
    else:
        # if neither are true, the user inputted a z-number, and we just extract that one
        image_slice, shape = czi.read_image(C=args.c_number, Z=args.z_number)
        image_slice = image_slice[0,0,0,0,0,:,:]
        img_extracted = image_slice

elif image_type == "tif":
    image = io.imread(filename)
    img_extracted = image[:,:,args.c_number]
    args.z_number = 0



extracted_filename =os.path.join(args.out_dir, f"{filename_base}_c{args.c_number}_{'maxIP' if args.maxIP else f'z{args.z_number}'}.tif")
io.imsave(extracted_filename, img_extracted, check_contrast=False)
print(f"Extracted to {extracted_filename}")


# Quick and dirty function to create the neurons, meta and directory variables used by the rest of the COUNTEN script
def readCOUNTENimage(filename: str):
    neurons = io.imread(filename)
    meta = {"Name": os.path.basename(os.path.splitext(filename)[0])}
    directory = os.path.join(str(args.out_dir), f"result_{os.path.splitext(meta['Name'])[0]}")
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

if not args.tile_size:
    neurons, meta, directory = readCOUNTENimage(extracted_filename)
    processCOUNTENdata(neurons,meta,directory)
else:
    extracted_filename_base = os.path.splitext(os.path.basename(extracted_filename))[0]
    nr_images = globalTilingFunc(extracted_filename, args.tile_size[0], args.tile_size[1], image_prefix =  extracted_filename_base)
    for i in range(1, nr_images + 1):
        neurons, meta, directory = readCOUNTENimage(f"{extracted_filename_base}_tile{i}.tif")
        processCOUNTENdata(neurons,meta,directory)

