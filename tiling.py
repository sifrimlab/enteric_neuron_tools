import math
import os
from skimage import io
import cv2

'''
    Tiling has so far been implemented such that an optimal divisor is sought that brings the tile resolution as close to a user defined resolution
    A future way of doing it might be to find a "best practice" resolution for the entire pipeline and then add padding functionality to this, to force the tiling into a certain resolution.
    However, tileWriter still does support irregular tiles, it's just not nice to have irregular tiles for downstream functionality.
'''
def calculateOptimalTileSize(X: int, Y: int , target_X: int, target_Y: int):
    """Calculates the optimal tile size to cut the given image into to get tiles the size of target_X, target_Y

    Parameters
    ----------
    img_path : str
        Path to input image.
    target_X : int
        Desired X-resolution.
    target_Y : int
        Desired Y-resolution

    Returns
    -------
    int, int
        Returns optimal X and Y-resolutions to tile the input image in to get the target resolution, while retaining evenly sized tiles.
    """
        # find the optimal division to the rounded up coordinate
    optimal_x = findOptimalDivisor(roundUpTo100(X), target_X)
    optimal_y = findOptimalDivisor(roundUpTo100(Y), target_Y)
    grid_size_x = roundUpTo100(X) / optimal_x
    grid_size_y = roundUpTo100(Y) / optimal_y
    # print(f"optimal_x: {optimal_x} ; optimal_y: {optimal_y}")
    return int(optimal_x), int(optimal_y), int(grid_size_x), int(grid_size_y)
    
# Function to round the coördinates up to 100
def roundUpTo100(x):
    result = x if x % 100 == 0 else x + 100 - x % 100
    return result
def writeTiles(img, prefix,tile_size_x, tile_size_y, output_dir="./"):
    # Don't forget, cv2 works with shape = (y, x), meaning rows, columns
    img_shape=img.shape
    offset_x = tile_size_x
    offset_y = tile_size_y
    tile_size = (tile_size_x, tile_size_y)
    offset = (offset_x, offset_y)
    tile_name_list = []
    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            # Multiplier is used for calculating the tile number, it represents how many tiles will be created on the x-axis
            multiplier = (int(math.ceil(img_shape[1]/(offset[0] * 1.0))))
            cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
            tile_number = multiplier*int(i) + int(j) +1
            io.imsave(os.path.join(output_dir, f"{prefix}_tiled_{tile_number}.tif"), cropped_img)
            tile_name_list.append(os.path.join(output_dir, f"{prefix}_tiled_{tile_number}.tif"))
    return tile_name_list


def pad(img, expected_width=0, expected_height=0):
    #returns an openCV image, probably to be saved by imwrite by the caller

    #Note hare that i use ceil to round up, this might cause some unwanted behaviour in the future.
    currentWidth, currentHeight = img.shape
    target_width = roundUpTo100(currentWidth)
    target_height = roundUpTo100(currentHeight)
    
    widthToAdd = math.floor((target_width - currentWidth)/2)
    heightToAdd = math.floor((target_height - currentHeight)/2)
    #In case there is still a difference in pixels to add and the wanted resolution, add that difference to the right or top,
    # depending on whether the difference is in width or in height respectively
    differenceWidth = (target_width-currentWidth) - widthToAdd*2
    differenceHeight = (target_height-currentHeight) - heightToAdd*2

    paddedImage = cv2.cv2.copyMakeBorder(img, heightToAdd+differenceHeight, heightToAdd, widthToAdd, widthToAdd+differenceWidth, cv2.cv2.BORDER_CONSTANT)
    newHeight = paddedImage.shape[0]
    newWidth = paddedImage.shape[1]

    if expected_height == 0 or expected_width == 0:
        pass
    else:
        if expected_width != newWidth:
            print("Warning: Width of resulting padded image is not equal to the entered expected width")
        if expected_height != newHeight:
            print("Warning: Height of resulting padded image is not equal to the entered expected height")
    return paddedImage

    # image = sys.argv[1]
    # prefix = os.path.splitext(image)[0]
    # target_x = int(sys.argv[2])
    # target_y = int(sys.argv[3])

    # image_padded = pad(image, target_x,target_y)
    # cv2.imwrite(f"{prefix}_padded.tif", image_padded)



def findOptimalDivisor(number: int, target_quotient: int):
    """Finds the optimal int divisor for the given number that results in the quotient as close to the given quotient as possible

    Parameters
    ----------
    number : int
        The number that will be divided by the optimal divisor
    target_quotient : int
        The quotient that you want the result of the division to be as close to as possible

    Returns
    -------
    int
        Returns the result of the optimal divison.
    """
    divisors = [i for i in range(1,number) if number % i==0]
    quotients = [number/divisor for divisor in divisors]
    min_loss = min(quotients, key=lambda x:abs(x-target_quotient))
    return min_loss

def completeTiling(image, prefix, output_dir):
    input_shape_x, input_shape_y = image.shape
    optimal_tile_x, optimal_tile_y, grid_size_x, grid_size_y= calculateOptimalTileSize(input_shape_x,input_shape_y,2000,2000)
    padded_img =  pad(image)
    tile_name_list = writeTiles(padded_img, prefix, optimal_tile_x, optimal_tile_y, output_dir)
    return tile_name_list

if __name__=='__main__':
    image_path = "/media/tool/enteric_neurones/slidescanner/Slide1-9-1_Region0000_Channel555_Seq0003_extracted.tif"
    prefix = os.path.splitext(image_path)[0]
    input_image = io.imread(image_path)
    input_shape_x, input_shape_y = input_image.shape
    optimal_tile_x, optimal_tile_y, grid_size_x, grid_size_y= calculateOptimalTileSize(input_shape_x,input_shape_y,2000,2000)
    padded_img =  pad(input_image)
    writeTiles(padded_img, prefix, optimal_tile_x, optimal_tile_y)


