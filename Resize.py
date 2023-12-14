import numpy as np
from PIL import Image

img = Image.open("Input/Image/161_epaper.png")
target = (1872, 1404)

ratio = max(target[0] / img.width, target[1] / img.height)
size = (int(img.width * ratio), int(img.height * ratio))
print(size)
black = np.zeros((target[1], target[0], 3), dtype=np.float32)

img = img.resize(size)

if size[0] > target[0]:
    img = np.array(img, dtype=np.float32)
    start = (size[0] - target[0]) // 2
    end = start + target[0]
    for i in range(target[1]):
        for j in range(start, end):
            black[i, j - start] = img[i, j]
    img = Image.fromarray(np.uint8(black))
elif size[1] > target[1]:
    img = np.array(img, dtype=np.float32)
    start = (size[1] - target[1]) // 2
    end = start + target[1]
    for i in range(start, end):
        for j in range(target[0]):
            black[i - start, j] = img[i, j]
    img = Image.fromarray(np.uint8(black))

img.show()
img.save("Input/Image/161_2_epaper.bmp")

# img2 = cv2.imread('Input/Image/img161.jpg')
# size2 = (1872, 1404)
# imgR2 = cv2.resize(img2, size2)

# cv2.imshow('Resized Image', imgR2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
