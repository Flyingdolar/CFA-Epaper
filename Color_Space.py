import math as M
import numpy as np


def gamma_correction(imgRGB: np.ndarray, gamma: float, reverse=False) -> np.ndarray:
    """
    Do gamma correction on image

    ## Parameters

    * `imgRGB` [np.ndarray] - Image to be corrected
    * `gamma` [float] - Gamma value
    * `reverse` [bool] - Reverse gamma correction

    ## Returns

    * `imgCorrected` [np.ndarray] - Corrected image
    """
    # 1 - Setup
    imgCorrected = np.zeros_like(imgRGB)
