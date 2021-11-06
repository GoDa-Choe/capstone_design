"""
Multi-View Partial Point Clouds

The data structure will be:

data
├── MVP_Train_CP.h5
|    ├── incomplete_pcds (62400, 2048, 3)
|    ├── complete_pcds (2400, 2048, 3)
|    └── labels (62400,)
├── MVP_Test_CP.h5
|    ├── incomplete_pcds (41600, 2048, 3)
|    ├── complete_pcds (1600, 2048, 3)
|    └── labels (41600,)
└── MVP_ExtraTest_Shuffled_CP.h5
     ├── incomplete_pcds (59800, 2048, 3)
     └── labels (59800,)

for details data_structure.md

"""
import random

import torch
import numpy as np
import h5py
from pathlib import Path

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")


class MVP(torch.utils.data.Dataset):
    def __init__(self, is_train=True, shape_type: str = "complete",
                 *, partition_type: str = '8-axis',
                 root='./data/'):
        """
        :param shape_type: complete / partial / occluded
        :param partition_type: 8-axis / 6-plain
        """
        self.root = PROJECT_ROOT / root
        self.shape_type = shape_type

        if self.shape_type == "occluded":
            self.directory = self.root / 'partitioned'
            if is_train:
                self.file_path = self.directory / f"{partition_type}_Partitioned_MVP_Train_CP.h5"
            else:
                self.file_path = self.directory / f"{partition_type}_Partitioned_MVP_Train_CP.h5"

        else:  # not occluded
            self.directory = self.root / 'raw'

            if is_train:
                self.file_path = self.directory / 'MVP_Train_CP.h5'
            else:
                self.file_path = self.directory / 'MVP_Test_CP.h5'

        input_file = h5py.File(self.file_path, 'r')

        if shape_type == "complete":
            self.input_data = np.array(input_file['complete_pcds'])  # dim: (2400, 2048, 3) (x ,z ,y)
            self.labels = np.array([input_file['labels'][i] for i in range(0, len(input_file['labels']), 26)])
            self.ground_truth_data = self.input_data.copy()

        else:  # shape_type == "incomplete" or "occluded"
            self.input_data = np.array(input_file['incomplete_pcds'])  # dim: (62400, 2048, 3)
            self.labels = np.array(input_file['labels'])
            if shape_type == "incomplete":
                self.ground_truth_data = np.repeat(input_file['complete_pcds'], 26, axis=0)
            else:  # shape_type == "occluded"
                self.ground_truth_data = np.array(input_file['complete_pcds'])

        self.len = self.input_data.shape[0]

        input_file.close()

    def __len__(self):
        return self.len

    # Todo 1. tensor type validation check required...
    def __getitem__(self, index):
        input_data = torch.from_numpy(self.input_data[index])
        ground_truth = torch.from_numpy(self.ground_truth_data[index])
        label = torch.from_numpy(np.array(self.labels[index].astype('int64')))

        # return input_data, ground_truth, label
        return input_data, label, ground_truth


if __name__ == "__main__":
    partial_train_dataset = MVP(
        shape_type="partial",
        is_train=True,
        root='./data/')

    partial_train_loader = torch.utils.data.DataLoader(
        dataset=partial_train_dataset,
        batch_size=32,
        shuffle=True,
        # num_workers=NUM_WORKERS
    )

    print(len(partial_train_dataset.labels))
    print(len(partial_train_dataset.input_data))
    #
    # print(complete_train_dataset[2399][0])
    # print(complete_train_dataset[2399][1])
    #
    # for i in range(len(complete_train_dataset)):
    #     print(complete_train_dataset[i][-1])

    for j, (points, labels) in enumerate(partial_train_loader):
        print(points.shape)
        print(labels.shape)
        print(labels)

        print(points[1])
        break
