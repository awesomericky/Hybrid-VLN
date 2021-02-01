import time
import numpy as np
from os import listdir
from os.path import join
import cv2
import matplotlib.pyplot as plt


"""
result = []
results = []

start = time.time()

folder = 'depth_result/5q7pvUzZiYa'
files = listdir(folder)

for file_1 in files:
    file_name = join(folder, file_1)
    result.append(np.load(file_name)['depth'])


folder = 'navigable_result/5q7pvUzZiYa'
files = listdir(folder)

for file_1 in files:
    file_name = join(folder, file_1)
    results.append(np.load(file_name)['navigable'])

end = time.time()
print(end-start)
print(len(files))
"""

cv2.namedWindow('depth')

depth = np.load('depth_result/5q7pvUzZiYa/6d11ca4d41e04bb1a725c2223c36b2aa.npz')['depth']
depth = np.squeeze(depth)/np.max(depth)
print(depth)
print(type(depth))
# while True:
# cv2.imwrite('depth1.png', depth)
# k = cv2.waitKey(1)

plt.imshow(depth, cmap='jet')
plt.savefig('depth1.png')
