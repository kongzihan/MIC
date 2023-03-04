from PIL import Image
import numpy as np
import random

image_name_list2 = ['test2.jpg']

for image_name in image_name_list2:
    image_path2 = r'../../' + image_name
    image_instance = Image.open(image_path2)
    # image_instance.load()
    # print(image_instance.size)

mask = np.ones([image_instance.size[1], image_instance.size[0]], dtype=np.uint8)
x1 = random.randint(0, image_instance.size[1] - 50)
y1 = random.randint(0, image_instance.size[0] - 50)
mask[x1:x1 + 50, y1:y1 + 50] = 0

array1 = np.array(image_instance)
array1 = np.transpose(array1, [2, 0, 1])
array1 = array1 * mask
array1 = np.transpose(array1, [1, 2, 0])

image_process = Image.fromarray(array1, mode='RGB')
image_process.show()

# arr1 = np.zeros([3, 40, 50], dtype=np.uint8)
# arr2 = np.ones([40, 50], dtype=np.uint8)
# arr3 = arr1 * arr2



