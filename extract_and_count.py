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

# First extract the correct image from the czi stack

ap = argparse.ArgumentParser(description="Extract tiff from a .czi image and count neurons and ganglia.")
ap.add_argument('czi_path',type=str,help="Path (relative or absolute) to target image")
ap.add_argument('-o', '--out_dir', type=str, help="Root directory where output should be stored, default is base dir of the input image")
ap.add_argument('-c', '--c_number', default=0, type=int, help="indexes (start at 0) of the channel contains the marker of interest. Default = 0.")

# Parameters

ap.add_argument('-p', '--pixel_density', default=3.2, type=float, help="Pixel density of the image (pixels/micrometer. default = 3.2")
ap.add_argument('-s', '--sigma', default=7, type=int, help="Sigma used to smooth the image using a gaussian smoother. default = 7")
ap.add_argument('-m', '--min_samples', default=2, type=int, help="Minimum number of neurons in a ganglion. default = 2")

ap.add_argument('-i', '--maxIP', default=True,action="store_false", help="Flag that turns off taking the maxIP of the z-dimension. If used, requires the usage of --z_number")
ap.add_argument('-z', '--z_number', default=None, type=int, help="index (start at 0) of the z-stack that needs to be extracted.")

args = ap.parse_args()

# if no out_dir is given, take the base dir of the input image
if args.out_dir is None:
    args.out_dir = os.path.dirname(args.czi_path)

extract_most_in_focus = bool(not args.maxIP and args.z_number is None)
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



if args.z_number and args.maxIP:
    ap.error('You cannot both take a maxIP and extract a single z-stack')

filename = args.czi_path
filename_base = os.path.splitext(os.path.basename(filename))[0]

# make output_dir if it doesn't exist
os.makedirs(args.out_dir, exist_ok=True)


czi = aicspylibczi.CziFile(filename)

z_min, z_max = czi.get_dims_shape()[0]['Z']
z_numbers = range(z_min,z_max)

if args.maxIP:

    def maxIPstack(img_list):
        parsed_list = img_list
        parsed_list = [img if isinstance(img, np.ndarray) else io.imread(img) for img in img_list]
        # now all elements in parsed_list are ndarrays
        maxIP = np.maximum.reduce(parsed_list)
        return maxIP

    image_list = []
    for z_num in z_numbers:
        image_slice, shape = czi.read_image(C=args.c_number, Z=z_num)
        image_slice = image_slice[0,0,0,0,0,:,:]
        image_list.append(image_slice)
    img_extracted = maxIPstack(image_list)
elif extract_most_in_focus:
    image_list = []
    for z_num in z_numbers:
        image_slice, shape = czi.read_image(C=args.c_number, Z=z_num)
        image_slice = image_slice[0,0,0,0,0,:,:]
        image_list.append(image_slice)
    img_extracted, args.z_number =  getMostInFocusImage(image_list)
else:
    image_slice, shape = czi.read_image(C=args.c_number, Z=args.z_number)
    image_slice = image_slice[0,0,0,0,0,:,:]
    img_extracted = image_slice



print(f"Extracted to {filename_base}_c{args.c_number}_{'maxIP' if args.maxIP else f'z{args.z_number}'}.tiff")
io.imsave(f"{args.out_dir}/{filename_base}_c{args.c_number}_{'maxIP' if args.maxIP else f'z{args.z_number}'}.tiff", img_extracted)

### Analyzing

neurons = io.imread(f"{args.out_dir}/{filename_base}_c{args.c_number}_{'maxIP' if args.maxIP else f'z{args.z_number}'}.tiff")
meta = {"Name": os.path.splitext(f"{filename_base}_c{args.c_number}_{'maxIP' if args.maxIP else f'z{args.z_number}'}.tiff")[0]}
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
