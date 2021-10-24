import datetime
import os
import numpy as np
import sys
from tqdm import tqdm
import time

a = datetime.datetime.now()
# print(a.strftime("%Y%m%d_%H%M%S"))

# os.mkdir("./temp/")


temp = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
repeated = np.repeat(temp, 2, axis=0)
double_repeated = np.repeat(repeated, 2, axis=0)
print(temp.shape)
print(temp)
print("---------")
print(repeated.shape)
print(repeated)
print("---------")
print(double_repeated.shape)
print(double_repeated)

# for i in tqdm(range(100)):
#     time.sleep(0.01)
#     for j in range(100):
#         time.sleep(0.01)

print("-----------------")
a = [[1, 2], [3, 4]]

a.extend([[5, 6]])
print(a)
