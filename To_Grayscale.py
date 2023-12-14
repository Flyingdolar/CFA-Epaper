import os
import cv2
import time
import math as m
import numpy as np
import pandas as pd
import Functions as f
from PIL import Image
from scipy.signal import convolve2d, correlate2d
import cv2
import numpy as np
from PIL import Image


def convert_to_grayscale(file_name):
    img0 = Image.open(file_name + ".png")
    fm = np.array(img0, dtype=np.float32)
    rows, cols, chls = fm.shape
    gm = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            gm[i, j] = np.max(fm[i, j])

    img1 = Image.fromarray(np.uint8(gm))
    # img1.show()
    img1.save(file_name + "_Gray.png")

    png_image = cv2.imread(file_name + "_Gray.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(file_name + "_Gray.bmp", png_image)
