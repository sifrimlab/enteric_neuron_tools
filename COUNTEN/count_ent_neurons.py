import os
import sys
import warnings
import importlib.util # This is to manually import the scripts to make sure it's all correct
warnings.filterwarnings('ignore')
import javabridge
import bioformats

# IO
image = "../imgs//1_ME_distal_20X_12WKJune2021_img1_c0_maxIP.tiff"
output_dir = "../output/"
script_path = "scripts/"

# Parameters
pixel_density = 3.2
sigma = 7
min_samples = 2

spec = importlib.util.spec_from_file_location("io",os.path.join(script_path, "io.py"))
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)







