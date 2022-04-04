import os
import  numpy as np
import  matplotlib.pyplot as plt
from skimage import io
from glob import glob

def labelTrainingSet(glob_pattern, out_dir:str = "", n_images =0) -> str:
    # X and Y are retrieved using famous kaggle facial expression dataset
    label_map=["g","b"]
    # x = glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Good/Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_555_tiles/*tile*.tif") 
    if not isinstance(glob_pattern, list):
        x = glob(glob_pattern) 
    else:
        x = glob_pattern

    if not out_dir:
        out_dir = os.path.dirname(x[0])
    # x = x[50:]
    good_dir = os.path.join(out_dir, "good/")
    bad_dir = os.path.join(out_dir, "bad/")
    os.makedirs(out_dir, exist_ok = True)
    os.makedirs(good_dir, exist_ok = True)
    os.makedirs(bad_dir, exist_ok = True)


    label_list = []
    for i, image_path in enumerate(x):
        plt.ion()
        plt.figure()
        image = io.imread(image_path)
        plt.imshow(image,cmap='gray')
        plt.title(os.path.basename(image_path))
        plt.show()
        label = input("input 'g' for good or 'b' for bad:")
        plt.close()
        label_list.append(label)
        if label == "g":
            io.imsave(os.path.join(good_dir, os.path.basename(image_path)),image)
        elif label == "b":
            io.imsave(os.path.join(bad_dir, os.path.basename(image_path)),image)

        # if a n_images was inputted and it isn't 0 obviously, check for a break.
        # Otherwise no break and all images are labeled
        if i >= n_images and n_images !=0 :
            break

    return out_dir
