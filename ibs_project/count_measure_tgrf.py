import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_holes
from pyxelperfect.segment import stardistSegment, otsuSegment
from pyxelperfect.display import showSegmentation
from pyxelperfect.measure import measureLabeledImage
from pyxelperfect.manipulate import automaticBrightnessAndContrast
from scipy import ndimage as ndi

# note to self: at this moment it seems the mast cell boundaries are more error-prone than DAPI, we match the mast centers to dapi labels
#and measure the boundaries of those matches in the mast labeled image, not the other way around. That decreases the number of matches, but probably in a correct way

filepath  = "../original_imgs/1391_C109_40X_BSA5_3.tif"
file_base = os.path.splitext(filepath)[0]
image = io.imread(filepath)

# Extract correct channels and take maxIP
mast = np.amax(image[:,:,:,0], axis=0) 
dapi = np.amax(image[:,:,:,1], axis=0) 
tgrf = np.amax(image[:,:,:,2], axis=0) 

# DAPI  segmentation
labeled_dapi = stardistSegment(dapi)
dapi_df = measureLabeledImage(labeled_dapi)

# Filter noise on Mast channel
mast_denoised = ndi.median_filter(mast, size=3)
mast_hole_mask = remove_small_holes(mast_denoised) # returns a mast for some reason
mast_denoised = mast_denoised * mast_hole_mask

labeled_mast = otsuSegment(mast_denoised) # thresholding works better on this low quality image, less distracted by noise
mast_df =  measureLabeledImage(labeled_mast)

# match_df = dapi_df.loc[labeled_mast[dapi_df['center_Y'], dapi_df['center_X']] != 0]

# Check for every mast cell if its center is labeled in dapi
match_df = mast_df.loc[labeled_dapi[mast_df['center_Y'], mast_df['center_X']] != 0]


intensity_col = []
tgrf_pos_col = []
avg_tgfr = np.mean(tgrf)
# For each mast that is also nucleated, get its label, calculate the mean intensity of all pixels that have that label, and check it it's larger than the average intensity of the image
for row in match_df.itertuples():
    mast_label_of_match = labeled_mast[match_df['center_Y'], match_df['center_X']]
    mean_int = np.mean(tgrf[labeled_mast == row.image_label])
    intensity_col.append(mean_int)
    tgrf_pos_col.append(mean_int > avg_tgfr)
match_df["mean tgrf expression"] = intensity_col
match_df["tgrf positive"] = tgrf_pos_col

match_df.to_csv(f"{file_base}.csv")

# Plot results
fig, axs = plt.subplots(2,3)
axs[0,0].imshow(dapi)
axs[0,0].set_title("original dapi")

axs[0,1].imshow(automaticBrightnessAndContrast(mast))
axs[0,1].set_title("original mast (enhanced contrast)")
axs[0,2].imshow(automaticBrightnessAndContrast(tgrf))
axs[0,2].set_title("original tgfr (enhanced contrast)")
axs[1,0].imshow(labeled_dapi)
axs[1,0].set_title("segmented dapi")
axs[1,1].imshow(labeled_mast)
axs[1,1].set_title("segmented mast")

for row in mast_df.itertuples():
    if row.image_label not in list(match_df['image_label']):
        labeled_mast[labeled_mast == row.image_label] = 0

axs[1,2].imshow(labeled_mast)
axs[1,2].set_title("nucleated mast")
for ax in axs.flatten():
    ax.axis("off")
plt.savefig(f"{file_base}_results.png")


