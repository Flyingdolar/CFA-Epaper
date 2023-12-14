import time
import math as M
import numpy as np
import Functions as f
import Color_Space_Change as csc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def bilateral_filter(
    img: np.ndarray, block_size: int, sigma_d: float, sigma_r: float
) -> np.ndarray:
    """
    Do bilateral filter on image with custom block size, sigma_d and sigma_r

    ## Parameters

    * `img` [np.ndarray] - Image to be filtered, should be one channel only
    * `block_size` [int] - Size of block, must be odd number
    * `sigma_d` [float] - Standard deviation of spatial distance
    * `sigma_r` [float] - Standard deviation of intensity distance

    ## Returns

    * `imgFiltered` [np.ndarray] - Filtered image
    """
    # Check if image is one channel
    if len(img.shape) != 2:
        raise ValueError("Image should be one channel only")

    # 1 - Setup
    imgFilt = np.zeros_like(img)  # filtered image
    pad = block_size // 2  # padding size

    # 2 - Pad image
    imgPad = np.pad(img, pad_width=pad, mode="symmetric")

    # 3 - Loop over image (Use tqdm to show progress)
    for hdx in tqdm(range(pad, img.shape[0] + pad)):
        for wdx in range(pad, img.shape[1] + pad):
            sumWeight, normVector = 0, 0
            # Calculate weight from block
            for bHdx in range(-pad, pad + 1):
                for bWdx in range(-pad, pad + 1):
                    # A - Spatial Distance Weight
                    spatialDist = M.sqrt(bHdx**2 + bWdx**2)
                    spatialWeight = M.exp(-(spatialDist**2) / (2 * sigma_d**2))
                    # B - Intensity Distance Weight
                    intensityDist = imgPad[hdx, wdx] - imgPad[hdx + bHdx, wdx + bWdx]
                    intensityWeight = M.exp(-(intensityDist**2) / (2 * sigma_r**2))
                    # C - Combine Weight and Sum
                    weight = spatialWeight * intensityWeight
                    sumWeight += weight * imgPad[hdx + bHdx, wdx + bWdx]
                    normVector += weight
            # 4 - Normalize and Assign
            imgFilt[hdx - pad, wdx - pad] = sumWeight / normVector
    return imgFilt


# <------------------------------------ Main ------------------------------------> #
if __name__ == "__main__":
    # 1 - Load Image
    imgFile = "Input/Image/img161.jpg"
    img = np.array(Image.open(imgFile))

    # 2 - Gamma Reverse
    imgLinear = f.sRGB_to_linearRGB(img)

    # 3 - Convert Color Spaces
    # Case A - Convert to LAB
    imgLAB = np.zeros_like(img)
    for hdx in range(img.shape[0]):
        for wdx in range(img.shape[1]):
            imgLAB[hdx, wdx] = csc.RGB_to_LAB(img[hdx, wdx])
    # Case B - Convert to XYZ
