import h5py
from pathlib import Path
from matplotlib import pyplot as plt

from src.dataset.category import CATEGORY

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")

INDEX = 10426


def visualize_generated_and_ground_truth(generated, ground_truth, label, fig_size):
    fig = plt.figure(figsize=fig_size)
    visualize_sub_plot(generated, fig, title="Generated",
                       row=1, col=2, index=1, cmap='winter')
    visualize_sub_plot(ground_truth, fig, title=f"GT: {CATEGORY[label]}",
                       row=1, col=2, index=2, cmap='autumn')
    fig.show()
    pass


def visualize_sub_plot(point_cloud, fig, title, row, col, index, *, cmap='winter'):
    sub = fig.add_subplot(row, col, index, projection="3d")
    sub.scatter(point_cloud[:, 0], point_cloud[:, 2], point_cloud[:, 1], c=point_cloud[:, 0], cmap=cmap)
    plt.title(title, fontdict={"fontsize": 50, "fontweight": "bold"})


def visualization_occluded_pcds(input_file, index):
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
    file_name = '8-axis_Partitioned_Reduced_MVP12_Train_CP.h5'

    input_file = h5py.File(file_path / file_name, 'r')
    visualization_occluded_pcds(input_file, INDEX)
