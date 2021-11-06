import h5py
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import time

from src.dataset.category import CATEGORY

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")

INDEX = 26


def visualization(input_file, index):
    incomplete_pcds = input_file["incomplete_pcds"][index:index + 8]
    complete_pcd = input_file["complete_pcds"][index]
    label = input_file["labels"][index]
    fig = plt.figure(figsize=(30, 30))

    for i in range(8):
        sub = fig.add_subplot(3, 3, i + 1, projection="3d")
        sub.scatter(incomplete_pcds[i][:, 0], incomplete_pcds[i][:, 2], incomplete_pcds[i][:, 1],
                    c=incomplete_pcds[i][:, 0], cmap='winter')

    sub = fig.add_subplot(3, 3, 9, projection="3d")
    sub.scatter(complete_pcd[:, 0], complete_pcd[:, 2], complete_pcd[:, 1], c=complete_pcd[:, 0], cmap="autumn")
    plt.title(f'Complete: {CATEGORY[label]}', fontdict={"fontsize": 50, "fontweight": "bold"})
    plt.show()


if __name__ == "__main__":
    file_path = PROJECT_ROOT / 'data' / 'partitioned'
    file_name = '8-axis_Partitioned_MVP_Train_CP.h5'

    input_file = h5py.File(file_path / file_name, 'r')
    visualization(input_file, INDEX)
