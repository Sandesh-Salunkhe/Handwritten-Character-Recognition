import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

path = r'E:\RawatTech\Alpha\segments_rno_last'
#print(classes)
#checking datafor img in os.listdir(path):



for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_array, cmap='gray')
    plt.show()
    break


img_size = 28

new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap='gray')
plt.show()

#full data loading

data =[]

for img in os.listdir(path):
    img_path = os.path.join(path, img)
    arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(arr, (28,28))
    data.append(new_arr)


len(data)

final=[]

for img in data:
    bitwiseNot = cv2.bitwise_not(img)
    final.append(bitwiseNot)


test = np.array(final).reshape(-1, img_size, img_size, 1)
print(test.shape)

np.savez_compressed("test_alpha.npz",test)

