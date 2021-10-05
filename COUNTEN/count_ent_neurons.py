# info: pixel width and height = 0.4151329 microns
# 1024 / 425.1 = 2.40884497765
import os
import sys
import warnings
from skimage import io
import importlib.util # This is to manually import the scripts to make sure it's all correct
warnings.filterwarnings('ignore')
import javabridge
import bioformats
import matplotlib.pyplot as plt
import time

# IO
image = sys.argv[1]
output_dir = sys.argv[2]
script_path = os.path.join(os.path.dirname(sys.argv[0]),"scripts/")

# make output_dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parameters
pixel_density = 3.2
sigma = 7
min_samples = 2

# spec = importlib.util.spec_from_file_location("io",os.path.join(script_path, "custom_io.py"))
# custom_io = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(custom_io)

# javabridge.start_vm(class_path=bioformats.JARS)

# neurons, directory, meta = custom_io.load_TIFF(image, output_dir)
# # print(neurons.type)

# javabridge.kill_vm()
neurons = io.imread(image)
meta = {"Name": os.path.splitext(image)[0]}
directory = os.path.join(str(output_dir), f"result_{os.path.splitext(meta['Name'])[0]}_{time.strftime('%m'+'_'+'%d'+'_'+'%Y')}")

# import another module
spec = importlib.util.spec_from_file_location("processing",os.path.join(script_path,"processing.py"))
processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processing)

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
spec = importlib.util.spec_from_file_location("analysis",os.path.join(script_path,"analysis.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

df, dist = analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)
