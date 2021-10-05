import sys
import os
import argparse
import importlib.util # This is to manually import the scripts to make sure it's all correct
import matplotlib.pyplot as plt
import numpy as np
import aicspylibczi
from skimage import io

# First extract the correct image from the czi stack

ap = argparse.ArgumentParser(description="Extract tiff from a .czi image and count neurons and ganglia.")
ap.add_argument('czi_path',type=str,help="Path (relative or absolute) to target image")
ap.add_argument('-o', '--out_dir', type=str, help="Root directory where output should be stored, default is base dir of the input image")
ap.add_argument('-c', '--c_number', default=0, type=int, help="indexes (start at 0) of the channel contains the marker of interest. Default = 0.")

# Parameters

ap.add_argument('-p', '--pixel_density', default=3.2, type=float, help="Pixel density of the image (pixels/micrometer. default = 3.2")
ap.add_argument('-s', '--sigma', default=7, type=int, help="Sigma used to smooth the image using a gaussian smoother. default = 7")
ap.add_argument('-m', '--min_samples', default=2, type=int, help="Minimum number of neurons in a ganglion. default = 2")

args = ap.parse_args()

# if no out_dir is given, take the base dir of the input image
if args.out_dir is None:
    args.out_dir = os.path.dirname(args.czi_path)

filename = args.czi_path
filename_base = os.path.splitext(os.path.basename(filename))[0]



czi = aicspylibczi.CziFile(filename)

z_min, z_max = czi.get_dims_shape()[0]['Z']
z_numbers = range(z_min,z_max)

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

print(f"Extracted {filename_base}_c{args.c_number}_maxIP.tiff")
io.imsave(f"{args.out_dir}/{filename_base}_c{args.c_number}_maxIP.tiff", img_extracted)

### Analyzing

script_path = os.path.join(os.path.dirname(sys.argv[0]),"scripts/")

# make output_dir if it doesn't exist
os.makedirs(args.out_dir, exist_ok=True)

# Load module
spec = importlib.util.spec_from_file_location("io",os.path.join(script_path, "io.py"))
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)

javabridge.start_vm(class_path=bioformats.JARS)

neurons, directory, meta = io.load_TIFF(f"{args.out_dir}/{filename_base}_c{args.c_number}_maxIP.tiff", args.out_dir)

javabridge.kill_vm()

# Load module
spec = importlib.util.spec_from_file_location("processing",os.path.join(script_path,"processing.py"))
processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processing)

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

# Load module
spec = importlib.util.spec_from_file_location("analysis",os.path.join(script_path,"analysis.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

# Run dataframe function from module
df, dist = analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)
