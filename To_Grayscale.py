import os
import cv2
import time
import math as m
import numpy as np
import pandas as pd
import Functions as f
from PIL import Image
from scipy.signal import convolve2d, correlate2d

fileName = "Output/CCMR_CFA"

img0 = Image.open(fileName + ".png")
fm = np.array(img0, dtype=np.float32)
rows, cols, chls = fm.shape
gm = np.zeros((rows, cols))

for i in range(rows):
    for j in range(cols):
        gm[i, j] = np.max(fm[i, j])

img1 = Image.fromarray(np.uint8(gm))
# img1.show()
img1.save(fileName + "_Gray.png")

png_image = cv2.imread(fileName + "_Gray.png", cv2.IMREAD_GRAYSCALE)
cv2.imwrite(fileName + "_Gray.bmp", png_image)
