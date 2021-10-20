import h5py
import numpy
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot

title_font = {
    'fontsize': 50,
    'fontweight': 'bold'
}

file_path = './data/MVP_Train_CP.h5'

input_file = h5py.File(file_path, 'r')

index = 4000
temp = index // 26

incomplete_pcd = np.array(input_file['incomplete_pcds'][index])
complete_pcd = np.array(input_file['complete_pcds'][temp])
label = np.array(input_file['labels'][index])

target_pcd = incomplete_pcd
label_pcd = incomplete_pcd
label_name = "incomplete"

print(complete_pcd.shape)

partial_ppp = [point for point in target_pcd if point[0] >= 0 and point[1] >= 0 and point[2] >= 0]
partial_npp = [point for point in target_pcd if point[0] < 0 and point[1] >= 0 and point[2] >= 0]
partial_pnp = [point for point in target_pcd if point[0] >= 0 and point[1] < 0 and point[2] >= 0]
partial_ppn = [point for point in target_pcd if point[0] >= 0 and point[1] >= 0 and point[2] < 0]
partial_pnn = [point for point in target_pcd if point[0] >= 0 and point[1] < 0 and point[2] < 0]
partial_npn = [point for point in target_pcd if point[0] < 0 and point[1] >= 0 and point[2] < 0]
partial_nnp = [point for point in target_pcd if point[0] < 0 and point[1] < 0 and point[2] >= 0]
partial_nnn = [point for point in target_pcd if point[0] < 0 and point[1] < 0 and point[2] < 0]

partial_ppp = np.array(partial_ppp)
partial_npp = np.array(partial_npp)
partial_pnp = np.array(partial_pnp)
partial_ppn = np.array(partial_ppn)
partial_pnn = np.array(partial_pnn)
partial_npn = np.array(partial_npn)
partial_nnp = np.array(partial_nnp)
partial_nnn = np.array(partial_nnn)

fig = pyplot.figure(figsize=(30, 30))

print(
    partial_ppp.shape[0]
    + partial_npp.shape[0]
    + partial_pnp.shape[0]
    + partial_ppn.shape[0]
    + partial_pnn.shape[0]
    + partial_npn.shape[0]
    + partial_nnp.shape[0]
    + partial_nnn.shape[0]
)

if partial_ppp.ndim == 2:
    ppp = fig.add_subplot(3, 3, 1, projection="3d")
    ppp.scatter(partial_ppp[:, 0], partial_ppp[:, 2], partial_ppp[:, 1], c='royalblue')
    pyplot.title("ppp", fontdict=title_font)

if partial_npp.ndim == 2:
    npp = fig.add_subplot(3, 3, 2, projection="3d")
    npp.scatter(partial_npp[:, 0], partial_npp[:, 2], partial_npp[:, 1], c='royalblue')
    pyplot.title("npp", fontdict=title_font)

if partial_pnp.ndim == 2:
    pnp = fig.add_subplot(3, 3, 3, projection="3d")
    pnp.scatter(partial_pnp[:, 0], partial_pnp[:, 2], partial_pnp[:, 1], c='royalblue')
    pyplot.title("pnp", fontdict=title_font)

if partial_ppn.ndim == 2:
    ppn = fig.add_subplot(3, 3, 4, projection="3d")
    ppn.scatter(partial_ppn[:, 0], partial_ppn[:, 2], partial_ppn[:, 1], c='royalblue')
    pyplot.title("ppn", fontdict=title_font)

if partial_pnn.ndim == 2:
    pnn = fig.add_subplot(3, 3, 5, projection="3d")
    pnn.scatter(partial_pnn[:, 0], partial_pnn[:, 2], partial_pnn[:, 1], c='royalblue')
    pyplot.title("pnn", fontdict=title_font)

if partial_npn.ndim == 2:
    npn = fig.add_subplot(3, 3, 6, projection="3d")
    npn.scatter(partial_npn[:, 0], partial_npn[:, 2], partial_npn[:, 1], c='royalblue')
    pyplot.title("npn", fontdict=title_font)

if partial_nnp.ndim == 2:
    nnp = fig.add_subplot(3, 3, 7, projection="3d")
    nnp.scatter(partial_nnp[:, 0], partial_nnp[:, 2], partial_nnp[:, 1], c='royalblue')
    pyplot.title("nnp", fontdict=title_font)

if partial_nnn.ndim == 2:
    nnn = fig.add_subplot(3, 3, 8, projection="3d")
    nnn.scatter(partial_nnn[:, 0], partial_nnn[:, 2], partial_nnn[:, 1], c='royalblue')
    pyplot.title("nnn", fontdict=title_font)

if label_pcd.ndim == 2:
    label_plot = fig.add_subplot(3, 3, 9, projection="3d")
    label_plot.scatter(label_pcd[:, 0], label_pcd[:, 2], label_pcd[:, 1], c='royalblue')
    pyplot.title(label_name, fontdict=title_font)

pyplot.show()
#
# #
# #
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.scatter(incomplete_pcd[:, 0], incomplete_pcd[:, 2], incomplete_pcd[:, 1], c='red')
# pyplot.show()
#
# print(incomplete_pcd.shape)
# print(complete_pcd.shape)
#
# for index in range(100):
#     temp = index // 26
#     incomplete_pcd = np.array(input_file['incomplete_pcds'][index])
#     complete_pcd = np.array(input_file['complete_pcds'][temp])
#
#     print(incomplete_pcd.shape, complete_pcd.shape)
