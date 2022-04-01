import os
import  numpy as np
import  matplotlib.pyplot as plt
from skimage import io
from glob import glob

# X and Y are retrieved using famous kaggle facial expression dataset
label_map=["g","b"]
x = glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Good/Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_555_tiles/*tile*.tif") 
x = x[100:]

out_dir = "./labeled_imgs/train/"
good_dir = os.path.join(out_dir, "good/")
bad_dir = os.path.join(out_dir, "bad/")
os.makedirs(out_dir, exist_ok = True)
os.makedirs(good_dir, exist_ok = True)
os.makedirs(bad_dir, exist_ok = True)


label_list = []
for i in range(len(x)):
    plt.ion()
    plt.figure()
    image = io.imread(x[i])
    plt.imshow(image,cmap='gray')
    plt.title(x[i])
    plt.show()
    label = input("input 'g' for good or 'b' for bad:")
    plt.close()
    label_list.append(label)
    if label == "g":
        io.imsave(os.path.join(good_dir, os.path.basename(x[i])),image)
    elif label == "b":
        io.imsave(os.path.join(bad_dir, os.path.basename(x[i])),image)
        
