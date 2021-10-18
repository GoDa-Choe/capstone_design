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

import torch
import numpy as np
import h5py
from pathlib import Path


class MVP(torch.utils.data.Dataset):
    def __init__(self, shape_type="complete", is_train=True, *, root='./data/'):
        if is_train:
            self.file_path = Path(root) / 'MVP_Train_CP.h5'
        else:
            self.file_path = Path(root) / 'MVP_Test_CP.h5'

        self.shape_type = shape_type

        input_file = h5py.File(self.file_path, 'r')
        if shape_type == "complete":
            self.input_data = np.array(input_file['complete_pcds'])  # dim: (2400, 2048, 3) (x ,y ,z)

        else:
            self.input_data = np.array(input_file['incomplete_pcds'])  # dim: (62400, 2048, 3)
        self.gt_data = self.input_data.copy()
        self.labels = np.array(input_file['labels'])

        self.len = self.input_data.shape[0]

        input_file.close()

    def __len__(self):
        return self.len

    # Todo 1. tensor type validation check required...
    def __getitem__(self, index):
        input_data = torch.from_numpy(self.input_data[index])
        groud_truth = torch.from_numpy(self.gt_data[index])

        if self.shape_type == "complete":
            label = torch.from_numpy(self.labels[index // 26])
        else:
            label = torch.from_numpy(self.labels[index])

        return input_data, groud_truth, label
