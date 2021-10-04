import sys
import os
from icecream import ic
import numpy as np
import aicspylibczi
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

filename = sys.argv[1]
filename_base = os.path.splitext(os.path.basename(filename))[0]

out_dir  =  os.path.dirname(filename)

czi = aicspylibczi.CziFile(filename)

# print(f"Dimensions of {filename_base}: {czi.get_dims_shape()[0]}")

z_min, z_max = czi.get_dims_shape()[0]['Z']
z_numbers = range(z_min,z_max)
c_min, c_max = czi.get_dims_shape()[0]['C']
c_numbers = range(c_min,c_max)

def maxIPstack(img_list):
    parsed_list = img_list
    parsed_list = [img if isinstance(img, np.ndarray) else io.imread(img) for img in img_list]
    # now all elements in parsed_list are ndarrays
    maxIP = np.maximum.reduce(parsed_list)
    return maxIP

image_list = []
for z_num in z_numbers:
        image_slice, shape = czi.read_image(C=0, Z=z_num)
        image_slice = image_slice[0,0,0,0,0,:,:]
        image_list.append(image_slice)
img_extracted = maxIPstack(image_list)

print(f"Saved {filename_base}_c0_maxIP.tiff")
io.imsave(f"{out_dir}/{filename_base}_c0_maxIP.tiff", img_extracted)


# plt.imshow(img_extracted)
# plt.axis('off')
# plt.show()

