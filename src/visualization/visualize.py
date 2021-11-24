import h5py
from pathlib import Path
from matplotlib import pyplot as plt

from src.dataset.category import CATEGORY

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.models.pcn import PCN
from src.models.pointnet import PointNetCls
from src.dataset.dataset import Partitioned_MVP

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
    plt.title(f'Complete:\n {CATEGORY[label]}', fontdict={"fontsize": 50, "fontweight": "bold"})
    plt.show()


def labeling(label, current_row):
    # if current_row == 0:
    #     plt.title(f'Complete', fontdict={"fontsize": 20})
    #     plt.title(f'{CATEGORY[label]}', fontdict={"fontsize": 20}, loc='right')
    # else:
    #     plt.title(f'{CATEGORY[label]}', fontdict={"fontsize": 20}, loc='right')
    plt.title(f'{CATEGORY[label]}', fontdict={"fontsize": 20}, loc='right', pad=0)


def get_generated_pcds(generator, incomplete_pcds):
    incomplete_pcds = torch.from_numpy(incomplete_pcds).to(device='cuda')

    generator.to(device='cuda')
    generator.eval()
    generated_pcds = generator(incomplete_pcds)['coarse_output']

    return generated_pcds.cpu().detach().numpy()


def visualization_generated_pcds(input_file, index_list, generator, num_incomplete=4):
    fig_row, fig_col = len(index_list), num_incomplete * 2 + 1
    fig = plt.figure(figsize=(30, 30))

    current_row = 0

    for index in index_list:
        incomplete_pcds = input_file["incomplete_pcds"][index:index + num_incomplete]
        generated_pcds = get_generated_pcds(generator, incomplete_pcds)
        complete_pcd = input_file["complete_pcds"][index]
        label = input_file["labels"][index]

        current_col = 1
        # incomplete_pcds draw
        for incomplete_pcd in incomplete_pcds:
            sub = fig.add_subplot(fig_row, fig_col, current_row + current_col, projection="3d")
            sub.scatter(incomplete_pcd[:, 0], incomplete_pcd[:, 2], incomplete_pcd[:, 1], c=incomplete_pcd[:, 0],
                        cmap='winter', s=1)
            if current_row == 0 and current_col == 1:
                plt.title(f'Occuluded', fontdict={"fontsize": 20})

            current_col += 1

        # generated_pcds draw
        for generated_pcds in generated_pcds:
            sub = fig.add_subplot(fig_row, fig_col, current_row + current_col, projection="3d")
            sub.scatter(generated_pcds[:, 0], generated_pcds[:, 2], generated_pcds[:, 1], c=generated_pcds[:, 0],
                        cmap='summer', s=1)
            if current_row == 0 and current_col == num_incomplete + 1:
                plt.title(f'Generated', fontdict={"fontsize": 20})

            current_col += 1

        # complete_pcd draw
        sub = fig.add_subplot(fig_row, fig_col, current_row + current_col, projection="3d")
        sub.scatter(complete_pcd[:, 0], complete_pcd[:, 2], complete_pcd[:, 1], c=complete_pcd[:, 0], cmap="autumn",
                    s=1)
        labeling(label, current_row)

        current_row += current_col

    plt.show()


if __name__ == "__main__":
    file_path = PROJECT_ROOT / 'data' / 'partitioned_mvp'
    file_name = 'Partitioned_MVP_Validation.h5'
    input_file = h5py.File(file_path / file_name, 'r')
    index_list = list(range(0, 3900 * 10, 3900))
    print(index_list)

    generator = PCN(emb_dims=1024, input_shape='bnc', num_coarse=2048, grid_size=4, detailed_output=False)
    # generator.load_state_dict(torch.load(PROJECT_ROOT / "pretrained_weights/partitioned_mvp/cd/20211123_092845/20.pth"))
    # generator.load_state_dict(torch.load(PROJECT_ROOT / "pretrained_weights/partitioned_mvp/ce/20211123_124301/20.pth"))
    generator.load_state_dict(
        torch.load(PROJECT_ROOT / "pretrained_weights/partitioned_mvp/ce_cd/20211123_212935/25.pth"))

    visualization_generated_pcds(input_file=input_file, index_list=index_list, generator=generator, num_incomplete=3)
