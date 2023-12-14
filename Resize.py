import numpy as np
from PIL import Image

def resize_image(input_path, output_path, target_width, target_height):
    img = Image.open(input_path)
    target = (target_width, target_height)

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
    img.save(output_path)

# # Example usage:
# input_path = "Input/Image/161_epaper.png"
# output_path = "Input/Image/161_2_epaper.bmp"
# target_width = 1872
# target_height = 1404

# resize_image(input_path, output_path, target_width, target_height)
