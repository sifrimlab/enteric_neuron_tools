import os
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage import feature
from skimage.morphology import dilation, disk
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage import filters
from skimage.transform import resize 
from skimage.color import rgb2gray
from scipy.ndimage.measurements import center_of_mass, label


def background_correct(img):

    gauss = gaussian(img, sigma=5)

    #size = 200
    #background = uniform_filter(gauss, size)
    #img_cor = img - background

    return(gauss)

def _binarization(image):
    data = image.ravel().reshape(1,-1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit(data.T)
    binary = kmeans.labels_.reshape(image.shape)
    #cluster can be "reverse" background = 1 and foreground = 0
    if np.count_nonzero(binary) > np.count_nonzero(1 - binary):
        binary = 1-binary
    # Adding some erosion to be more conservative
    #binary = morphology.opening(binary, morphology.ball(2))
    return(binary)

def _binarize(image):
    subsampled_image = subsample(image)
    thresh = filters.threshold_otsu(subsampled_image)
    binary = binarize(image, threshold=thresh)
    binary = dilation(binary, disk(10))
    return binary

def subsample(img):
    x,y=img.shape
    crop_x=int(x/3)
    crop_y=int(y/3)
    start_x=int((x/2)-(crop_x/2))
    start_y=int((y/2)-(crop_y/2))
    return img[start_x:start_x+crop_x, start_y:start_y+crop_y]


# You added output_filename to save the created image
def wide_clusters(img, sigma, pixel_density, min_samples,meta, directory, plot = True, save=False):
    # If the image is already single channel, this will throw an error
    try:
        grayscale = rgb2gray(img)
    except ValueError:
        grayscale = img

    gauss = gaussian(grayscale, sigma=sigma)
    img_subsampled = subsample(gauss)

    thresh =filters.threshold_otsu(img_subsampled)


    # Used to be without the tmp and masking, but skimage has removed the indices 
    tmp_is_peak = feature.peak_local_max(gauss, min_distance = int(2.5 * pixel_density), threshold_abs=thresh +
                                     (10*thresh)/100,
                                      exclude_border=False)
    is_peak = np.zeros_like(gauss, dtype=bool)
    is_peak[tuple(tmp_is_peak.T)] = True

    plabels = label(is_peak)[0]
    merged_peaks = center_of_mass(is_peak, plabels, range(1, np.max(plabels)+1))
    local_maxi = np.array(merged_peaks)

    X = local_maxi

    if len(X) > 0:
        # Compute DBSCAN
        db = DBSCAN(eps=20.6*pixel_density, min_samples=min_samples).fit(X)
        labels = db.labels_

        if plot:
            label_plot = np.copy(labels)
            label_plot[labels == 0] = max(labels)+1
            label_plot[labels == -1] = 0

            fig, (ax) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8),
                                   sharex=True, sharey=True, squeeze=True)
            ax[0].imshow(gauss, alpha=0.6)
            ax[0].scatter(local_maxi[:,1], local_maxi[:,0], s=4)
            ax[0].axis("off")

            ax[1].imshow(gauss, alpha=0.3)
            ax[1].scatter(local_maxi[:,1], local_maxi[:,0], c=label_plot, cmap = "nipy_spectral")
            ax[1].axis("off")
            if save:
                try:
                    filename = f"{os.path.splitext(meta['Name'])[0]}_clusters.tif"
                    plt.savefig(os.path.join(directory,filename))
                except IOError:
                    plt.savefig(filename)
    # If there are no elements in X, just plot the images anyway even though there is nothing to see, this way the output is consistent in looking over the tiles
    else:
        # Create an empty labels array to allow for the rest of the processing to continue
        labels = np.empty(0)
        fig, (ax) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8),
                       sharex=True, sharey=True, squeeze=True)

        ax[0].imshow(gauss, alpha=0.6)
        ax[0].axis("off")

        ax[1].imshow(gauss, alpha=0.3)
        ax[1].axis("off")
        try:
            filename = f"{os.path.splitext(meta['Name'])[0]}_clusters.tif"
            plt.savefig(os.path.join(directory,filename))
        except IOError:
            plt.savefig(filename)
    


    plt.close()
    return (local_maxi, labels, gauss)

def segmentation(img, local_maxi, labels, meta, directory, plot=True, save=False):

    only_clusters = np.zeros(img.shape, dtype=np.uint)
    for pos, new in zip(local_maxi, labels):
        if new > 0:
            only_clusters[int(pos[0]), int(pos[1])] = new
        elif new == 0:
            only_clusters[int(pos[0]), int(pos[1])] = max(labels) + 1
    only_clusters = dilation(only_clusters, disk(10))

    binary = _binarize(img)

    dist_water = ndi.distance_transform_edt(binary)
    segmentation_ws = watershed(-img, only_clusters, mask = binary)

    ganglion_prop = regionprops(segmentation_ws)

    if plot == True:
        if segmentation_ws.size > 250000000:
            x,y=img.shape #Array splitting
            img_1 = img[0:x, 0:int(y/2)]
            img_2 = img[0:x, int(y/2)+1:y]
            seg_ws1 = segmentation_ws[0:x, 0:int(y/2)]
            seg_ws2 = segmentation_ws[0:x, int(y/2)+1:y]
            seg_ws = [seg_ws1, seg_ws2]
            img_list = [img_1, img_2]

            for i in range(2):
                image_label_overlay = label2rgb(seg_ws[i], image=img_list[i].astype('uint16'), 
                                                bg_label=0)

                fig,ax = plt.subplots(1,1, figsize=(16,16))
                ax.imshow(image_label_overlay, interpolation='nearest')
                ax.axis('off')

                for prop in regionprops(seg_ws[i]):
                    ax.annotate(prop.label,
                                (prop.centroid[1]-5, prop.centroid[0]), color='green',
                                fontsize=8,weight = "bold")

                if save:
                    try:
                        filename = os.path.splitext(meta['Name'])[0]+str(i+1)+'.tif'
                        plt.savefig(os.path.join(directory,filename))
                    except IOError:
                        print(IOError)
                        plt.savefig(filename)

        else:
            image_label_overlay = label2rgb(segmentation_ws, image=img.astype('uint16'), 
                                                bg_label=0)

            fig,ax = plt.subplots(1,1, figsize=(16,16))
            ax.imshow(image_label_overlay, interpolation='nearest')
            ax.axis('off')

            for prop in ganglion_prop:
                ax.annotate(prop.label,
                                (prop.centroid[1]-5, prop.centroid[0]), color='green',
                                fontsize=8,weight = "bold")

            if save:
                try:
                    filename = f"{os.path.splitext(meta['Name'])[0]}_segmentation.tif"
                    plt.savefig(os.path.join(directory,filename), transparent=True)
                except IOError:
                    plt.savefig(filename, transparent=True)
 
    return ganglion_prop
