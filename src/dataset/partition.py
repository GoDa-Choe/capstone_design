import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import time

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")


class RawDatasetLoader:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.input_file = self.load()

        self.incomplete_pcds = self.input_file['incomplete_pcds']  # (62400, 2048, 3)
        self.complete_pcds = self.input_file['complete_pcds']  # (2400, 2048, 3)
        self.labels = self.input_file['labels']  # (62400,)

    def load(self, directory="data/raw/"):
        file_path = PROJECT_ROOT / directory / self.file_name
        input_file = h5py.File(file_path, 'r')
        print(f"{self.file_name} was loaded.\n")
        return input_file

    def close(self):
        self.input_file.close()


class Partition:
    def __init__(self, raw_dataset: RawDatasetLoader, partition_type="8-axis",
                 num_select: int = 1, num_points: int = 200):
        self.raw_dataset = raw_dataset

        self.partition_type = partition_type
        if self.partition_type == "8-axis":
            self.num_partition = 8
        else:
            self.num_partition = 6

        self.num_select = num_select
        self.num_points = num_points

        self.complete_pcds = np.repeat(self.raw_dataset.complete_pcds, 26 * self.num_select, axis=0)
        self.labels = np.repeat(self.raw_dataset.labels, self.num_select, axis=0)

        if self.partition_type == "8-axis":
            self.partitioned_pcds = self.eight_axis_partition()
        else:
            self.partitioned_pcds = self.six_plain_partition()
        self.raw_dataset.close()

        self.occluded_pcds = self.select_pcds_sample_points(self.partitioned_pcds)

    def sample_points(self, selected_pcds):
        for i in range(len(selected_pcds)):
            random.shuffle(selected_pcds[i])
            selected_pcds[i] = selected_pcds[i][:self.num_points]

    def select_pcds_sample_points(self, partitioned_pcds):

        print(f"Partitioned_pcds is being selected and sampled...")
        occluded_pcds = []
        for i in tqdm(range(0, len(partitioned_pcds), 26 * self.num_partition)):
            candidate_pcds = [partitioned_pcd for partitioned_pcd
                              in self.partitioned_pcds[i:i + 26 * self.num_partition]
                              if len(partitioned_pcd) >= self.num_points]

            random.shuffle(candidate_pcds)
            selected_pcds = candidate_pcds[:26 * self.num_select]
            self.sample_points(selected_pcds)

            occluded_pcds.extend(selected_pcds)

        print(f"{len(partitioned_pcds)} -> {len(occluded_pcds)}")
        print()
        return occluded_pcds

    def eight_axis_partition(self):
        print(f"{self.raw_dataset.file_name} is partitioning by 8-axis...")
        time.sleep(0.1)

        partitioned_pcds = []
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
            partitioned_pcds.extend([ppp, npp, pnp, ppn, pnn, npn, nnp, nnn])

        time.sleep(0.1)
        print(f"{self.raw_dataset.incomplete_pcds.shape} -> ({len(partitioned_pcds)}, x, 3)\n")

        return partitioned_pcds

    def statistics(self):
        with open(PROJECT_ROOT / f'data/partitioned/statistics_{self.partition_type}.txt', 'w') as file:
            file.write(f"xy_p xy_n yz_p yz_n zx_p zx_n\n")

            for index, partial_pcd in enumerate(self.partitioned_pcds, start=1):
                file.write(f"{len(partial_pcd)} ")

                if index % 6 == 0:
                    file.write("\n")

    def six_plain_partition(self):
        print(f"{self.raw_dataset.file_name} is partitioning by 6-plain...")
        time.sleep(0.1)

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

        time.sleep(0.1)
        print(f"{self.raw_dataset.incomplete_pcds.shape} -> ({len(partial_pcds)}, x, 3)\n")
        return partial_pcds

    def save(self, directory=None, file_name=None):
        if directory is None:
            directory = 'partitioned'
        file_path = PROJECT_ROOT / 'data' / directory

        if file_name is None:
            file_name = f"{self.partition_type}_Partitioned_{self.raw_dataset.file_name}"
        output_file = h5py.File(file_path / file_name, 'w')

        output_file.create_dataset('incomplete_pcds', data=self.occluded_pcds)
        output_file.create_dataset('complete_pcds', data=self.complete_pcds)
        output_file.create_dataset('labels', data=self.labels)
        output_file.close()

        print(f"{file_name} was successfully saved at {file_path}.\n")


if __name__ == "__main__":
    # raw = RawDatasetLoader(file_name="MVP_Train_CP.h5")
    raw = RawDatasetLoader(file_name="MVP_Test_CP.h5")

    partition = Partition(raw, partition_type="8-axis", num_select=1, num_points=200)
    partition.save()
