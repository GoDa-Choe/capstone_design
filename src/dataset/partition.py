import h5py
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot
from tqdm import tqdm

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design")


# noinspection SpellCheckingInspection
class RawDatasetLoader:
    def __init__(self, file_name: str, num_points, out_directory):
        self.file_name = file_name
        self.num_points = num_points

        self.input_file = self.load()

        self.incomplete_pcds = self.input_file['incomplete_pcds']  # (62400, 2048, 3)
        self.complete_pcds = np.repeat(self.input_file['complete_pcds'], 26, axis=0)  # (62400, 2048, 3)
        self.labels = self.input_file['labels']  # (62400,)

    def load(self, directory="data/raw/"):
        file_path = PROJECT_ROOT / directory / self.file_name
        input_file = h5py.File(file_path, 'r')
        print(f"{self.file_name} was loaded.")
        return input_file

    @staticmethod
    def draw_plot(*args, complete, label):
        fig = pyplot.figure(figsize=(30, 30))
        for i, points in enumerate(args, 1):
            if points.ndim == 2:
                sub = fig.add_subplot(3, 3, i, projection="3d")
                sub.scatter(points[:, 0], points[:, 2], points[:, 1], c='royalblue')

        sub = fig.add_subplot(3, 3, 9, projection="3d")
        sub.scatter(complete[:, 0], complete[:, 2], complete[:, 1], c='royalblue')
        pyplot.title(label, fontdict={"fontsize": 50})
        pyplot.show()


# noinspection SpellCheckingInspection,DuplicatedCode,PyChainedComparisons
class Partition:
    def __init__(self, raw_dataset: RawDatasetLoader, partition_type="6-plain", select_num=3, num_points=3):
        self.raw_dataset = raw_dataset
        self.partition_type = partition_type
        complete_pcds = np.repeat(self.raw_dataset.complete_pcds, 6, axis=0)
        labels = np.repeat(self.raw_dataset.labels, 6, axis=0)

        if partition_type == "6-plain":
            self.full_partial_pcds = self.six_plain_partition()
        else:
            self.full_partial_pcds = self.eight_axis_partition()

    def eight_axis_partition(self):
        print(f"{self.raw_dataset.file_name} is partitioning by 8-axis now.")
        partial_pcds = []
        for incomplete_pcd in tqdm(self.raw_dataset.incomplete_pcds):
            ppp, npp, pnp, ppn, pnn, npn, nnp, nnn = [], [], [], [], [], [], [], []
            for point in incomplete_pcd:
                if point[0] >= 0 and point[1] >= 0 and point[2] >= 0:
                    ppp.append(point)
                elif point[0] < 0 and point[1] >= 0 and point[2] >= 0:
                    npp.append(point)
                elif point[0] >= 0 and point[1] < 0 and point[2] >= 0:
                    pnp.append(point)
                elif point[0] >= 0 and point[1] >= 0 and point[2] < 0:
                    ppn.append(point)
                elif point[0] >= 0 and point[1] < 0 and point[2] < 0:
                    pnn.append(point)
                elif point[0] < 0 and point[1] >= 0 and point[2] < 0:
                    npn.append(point)
                elif point[0] < 0 and point[1] < 0 and point[2] >= 0:
                    nnp.append(point)
                elif point[0] < 0 and point[1] < 0 and point[2] < 0:
                    nnn.append(point)
                else:
                    print("fatal error detected!")
            partial_pcds.extend((ppp, npp, pnp, ppn, pnn, npn, nnp, nnn))

        print(f"{self.raw_dataset.incomplete_pcds.shape} -> ({len(partial_pcds)}, x, 3)")
        print(f"{self.raw_dataset.file_name} was partitionied by 8-axis.")

        return partial_pcds

    def statistics(self):
        with open(PROJECT_ROOT / f'data/partitioned/statistics_{self.partition_type}.txt', 'w') as file:
            file.write(f"partial_points cloud {self.partition_type} ({len(self.full_partial_pcds)}, x, 3)")

            for index, partial_pcd in enumerate(self.full_partial_pcds, start=1):
                file.write(f"{len(partial_pcd)} ")

                if index % 6 == 0:
                    file.write("/n")

    def six_plain_partition(self):
        print(f"{self.raw_dataset.file_name} is partitioning by 6-plain now.")
        partial_pcds = []
        for incomplete_pcd in tqdm(self.raw_dataset.incomplete_pcds):
            xy_p, xy_n, yz_p, yz_n, zx_p, zx_n = [], [], [], [], [], []
            for point in incomplete_pcd:
                if point[0] >= 0 and point[1] >= 0:
                    xy_p.append(point)
                elif point[0] < 0 and point[1] < 0:
                    xy_n.append(point)
                elif point[1] >= 0 and point[2] >= 0:
                    yz_p.append(point)
                elif point[1] < 0 and point[2] < 0:
                    yz_n.append(point)
                elif point[2] >= 0 and point[0] >= 0:
                    zx_p.append(point)
                elif point[2] < 0 and point[0] < 0:
                    zx_n.append(point)
                else:
                    print("fatal error detected!")

            partial_pcds.extend((xy_p, xy_n, yz_p, yz_n, zx_p, zx_n))

        print(f"{self.raw_dataset.incomplete_pcds.shape} -> ({len(partial_pcds)}, x, 3)")
        print(f"{self.raw_dataset.file_name} was partitionied by 6-plain.")
        return partial_pcds


if __name__ == "__main__":
    temp = RawDatasetLoader(file_name="MVP_Train_CP.h5", num_points=1000, out_directory="temp")
    print(temp.incomplete_pcds.shape)
    print(temp.complete_pcds.shape)
    print(temp.labels.shape)
    print("-----------")

    partial = Partition(temp, partition_type="6-plain")
    partial.statistics()
