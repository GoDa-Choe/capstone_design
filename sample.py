import h5py
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot

file_path = './data/MVP_Train_CP.h5'

input_file = h5py.File(file_path, 'r')

index = 5
temp = index // 26

incomplete_pcd = np.array(input_file['incomplete_pcds'][index])
complete_pcd = np.array(input_file['complete_pcds'][temp])
label = np.array(input_file['labels'][index])
print(len(input_file['labels']))

print(label)
fig = pyplot.figure(figsize=(20, 10))
ax = fig.add_subplot(121, projection="3d")
ax.scatter(complete_pcd[:, 0], complete_pcd[:, 2], complete_pcd[:, 1], c='royalblue')

#
#
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(incomplete_pcd[:, 0], incomplete_pcd[:, 2], incomplete_pcd[:, 1], c='red')
pyplot.show()

print(incomplete_pcd.shape)
print(complete_pcd.shape)

for index in range(100):
    temp = index // 26
    incomplete_pcd = np.array(input_file['incomplete_pcds'][index])
    complete_pcd = np.array(input_file['complete_pcds'][temp])

    print(incomplete_pcd.shape, complete_pcd.shape)
