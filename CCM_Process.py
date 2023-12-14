import time
import math as M
import numpy as np
import Functions as f
import Color_Space_Change as csc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
import math as M
import numpy as np
import Functions as f
import Color_Space_Change as csc
from PIL import Image
from tqdm import tqdm

# ----------------------------------- Matrix Result ----------------------------------- #
CCMList = np.array(
    [
        [  # Optimal Color Correction Matrix0:
            [2.23696157, -2.59225501, 1.35529344],
            [-0.34452285, 1.98257122, -0.63804837],
            [-0.7370915, -3.17873355, 4.91582504],
        ],
        [  # Optimal Color Correction Matrix1:
            [2.36368878, -1.45420957, 0.09052078],
            [-0.36681442, 2.52024649, -1.15343207],
            [-0.65268323, -2.52757087, 4.1802541],
        ],
        [  # Optimal Color Correction Matrix2:
            [2.49535239, -3.47214075, 1.97678836],
            [-0.7174657, 3.09011861, -1.3726529],
            [-1.43012356, -2.15842275, 4.58854631],
        ],
        [  # Optimal Color Correction Matrix3:
            [2.15829596, -2.52507097, 1.36677502],
            [-0.21226783, 1.65694319, -0.44467536],
            [-1.16075226, -4, 6.16075226],
        ],
    ]
)
# Center of M1, M2, M3
matPos = np.array([[0.241886, 0.211849], [0.444181, 0.354937], [0.356936, 0.482268]])

# ----------------------------------- Reverse Matrix ----------------------------------- #
# Reverse Matrix of CCMList


CCMListReverse = np.array(
    [
        # Optimal Color Correction Matrix0:
        # 7717787486617535300000000/15119761172303927450439631	 8434955360400538400000000/15119761172303927450439631	-1032981685043963100000000/15119761172303927450439631
        # 2163914082998019000000000/15119761172303927450439631	11995486973953472800000000/15119761172303927450439631	  960360124956036900000000/15119761172303927450439631
        # 2556482736443247500000000/15119761172303927450439631	 9021433926323088500000000/15119761172303927450439631	 3541844544956036900000000/15119761172303927450439631
        [
            [0.51044374, 0.55787623, -0.06831997],
            [0.14311827, 0.79336484, 0.06351688],
            [0.16908222, 0.59666511, 0.23425268],
        ],
        # Optimal Color Correction Matrix1:
        # 7619889422177308100000000/14919254453044319978226919	5850167830594058400000000/14919254453044319978226919	1449197276471847700000000/14919254453044319978226919
        # 2286203252177308100000000/14919254453044319978226919	9939901108791517400000000/14919254453044319978226919	2693150114937527000000000/14919254453044319978226919
        # 2572072062177308100000000/14919254453044319978226919	6923529105318349700000000/14919254453044319978226919	5423653311269382800000000/14919254453044319978226919
        [
            [0.51074197, 0.39212200, 0.09713604],
            [0.15323844, 0.66624650, 0.18051506],
            [0.17239950, 0.46406669, 0.36353380],
        ],
        # Optimal Color Correction Matrix2:
        # 448655483926574164000000/861570177731997143221577	466613346322157700000000/861570177731997143221577	-53698657182868184000000/861570177731997143221577
        # 210207513691155640000000/861570177731997143221577	571083664322157700000000/861570177731997143221577	 80278994007847160000000/861570177731997143221577
        # 238713828663205064000000/861570177731997143221577	414064626322157700000000/861570177731997143221577	208791718605988116000000/861570177731997143221577
        [
            [0.52074166, 0.54158484, -0.06232650],
            [0.24398188, 0.66284057, 0.09317755],
            [0.27706835, 0.48059303, 0.24233861],
        ],
        # Optimal Color Correction Matrix3:
        # 4214657531242054700000000/8688360016412168812420547	5044618302543946100000000/8688360016412168812420547	-570915859520407300000000/8688360016412168812420547
        #  911943721242054700000000/8688360016412168812420547	7441606953347707400000000/8688360016412168812420547	 334809332702969500000000/8688360016412168812420547
        # 1386185936242054700000000/8688360016412168812420547	5782082837543946100000000/8688360016412168812420547	1520091228764308650000000/8688360016412168812420547
        [
            [0.48509241, 0.58061801, -0.06571043],
            [0.10496155, 0.85650306, 0.03853539],
            [0.15954518, 0.66549761, 0.17495721],
        ],
    ]
)


# ----------------------------------- Test Process ----------------------------------- #
# # Test Matrix Multiplication is a Identity Matrix
# print("Test Matrix Multiplication is a Identity Matrix")
# for idx in range(4):
#     print(np.dot(CCMList[idx], CCMListReverse[idx]))



import matplotlib.pyplot as plt

def apply_color_correction(input_path, output_path):
    imgsRGB = np.array(Image.open(input_path), dtype=np.float32)
    imgRGB = np.zeros(imgsRGB.shape)

    # 1. Reverse Gamma Correction
    print("Reverse Gamma Correction")
    for row in tqdm(range(imgsRGB.shape[0])):
        for col in range(imgsRGB.shape[1]):
            imgRGB[row][col] = csc.sRGB_to_linearRGB(imgsRGB[row][col])

    # 2. Apply Color Correction Matrix 0 to all pixels
    imgRGB = np.dot(imgRGB, CCMListReverse[0])

    # 3. Create a corresponding image in the CIEYxy color space
    print("Create a corresponding image in the CIEYxy color space")
    imgYxy = np.zeros(imgRGB.shape)
    for row in tqdm(range(imgRGB.shape[0])):
        for col in range(imgRGB.shape[1]):
            imgYxy[row][col] = csc.XYZ_to_Yxy(*csc.RGB_to_XYZ(*imgRGB[row][col]))

    # 4. Apply CCM1/CCM2/CCM3 according to which xy color position is closer to the pixel
    print("Apply CCMs according to which xy color position is closer to the pixel")
    for row in tqdm(range(imgRGB.shape[0])):
        for col in range(imgRGB.shape[1]):
            xyDist = np.zeros(3)
            for idx in range(3):
                xyDist[idx] = M.sqrt(
                    M.pow(imgYxy[row][col][1] - matPos[idx][0], 2)
                    + M.pow(imgYxy[row][col][2] - matPos[idx][1], 2)
                )
            minIdx = np.argmin(xyDist) + 1
            imgRGB[row][col] = np.dot(imgRGB[row][col], CCMListReverse[minIdx])


    # 5. Convert the image back to the sRGB color space
    print("Convert the image back to the sRGB color space")
    for row in tqdm(range(imgRGB.shape[0])):
        for col in range(imgRGB.shape[1]):
            imgsRGB[row][col] = csc.linearRGB_to_sRGB(imgRGB[row][col])

    # 6. Output the image in the sRGB color space
    # Clip the value to [0, 255]
    print("Clip the value to [0, 255]")
    imgsRGB = np.clip(imgsRGB, 0, 255)
    plt.imshow(imgsRGB.astype(np.uint8))
    plt.show()

    # 7. Save the image
    print("Save the image")
    Image.fromarray(imgsRGB.astype(np.uint8)).save(output_path)

