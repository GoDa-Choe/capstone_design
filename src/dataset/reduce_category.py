import h5py
import numpy as np
from pathlib import Path

from src.dataset.partition import RawDatasetLoader

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")


# REDUCED CATEGORIES 0 / 12 / 14 / 15


class Reduced_MVP12(RawDatasetLoader):
    def __init__(self, file_name, is_train=True):
        super(Reduced_MVP12, self).__init__(file_name)
        self.is_train = is_train

        complete_indices = self.get_indices(is_complete=True)
        indices = self.get_indices(is_complete=False)

        self.incomplete_pcds = np.array(self.incomplete_pcds)
        self.complete_pcds = np.array(self.complete_pcds)
        self.labels = np.array(self.labels)

        self.reduced_complete_pcds = self.complete_pcds[complete_indices]
        self.reduced_incomplete_pcds = self.incomplete_pcds[indices]
        self.reduced_labels = self.labels[indices]

    def get_indices(self, is_complete=True):
        if is_complete:
            coefficient = 1
        else:
            coefficient = 26
        if self.is_train:
            cat0 = range(0 * coefficient, 200 * coefficient)
            cat1 = range(200 * coefficient, 400 * coefficient)
            cat2 = range(400 * coefficient, 600 * coefficient)
            cat3 = range(600 * coefficient, 800 * coefficient)
            cat4 = range(800 * coefficient, 1000 * coefficient)
            cat5 = range(1000 * coefficient, 1200 * coefficient)
            cat6 = range(1200 * coefficient, 1400 * coefficient)
            cat7 = range(1400 * coefficient, 1600 * coefficient)
            cat8 = range(1600 * coefficient, 1700 * coefficient)
            cat9 = range(1700 * coefficient, 1800 * coefficient)
            cat10 = range(1800 * coefficient, 1900 * coefficient)
            cat11 = range(1900 * coefficient, 2000 * coefficient)
            cat12 = range(2000 * coefficient, 2100 * coefficient)
            cat13 = range(2100 * coefficient, 2200 * coefficient)
            cat14 = range(2200 * coefficient, 2300 * coefficient)
            cat15 = range(2300 * coefficient, 2400 * coefficient)
        else:
            cat3 = range(0 * coefficient, 150 * coefficient)
            cat6 = range(150 * coefficient, 300 * coefficient)
            cat5 = range(300 * coefficient, 450 * coefficient)
            cat1 = range(450 * coefficient, 600 * coefficient)
            cat4 = range(600 * coefficient, 750 * coefficient)
            cat2 = range(750 * coefficient, 900 * coefficient)
            cat0 = range(900 * coefficient, 1050 * coefficient)
            cat7 = range(1050 * coefficient, 1200 * coefficient)
            cat8 = range(1200 * coefficient, 1250 * coefficient)
            cat9 = range(1250 * coefficient, 1300 * coefficient)
            cat10 = range(1300 * coefficient, 1350 * coefficient)
            cat11 = range(1350 * coefficient, 1400 * coefficient)
            cat12 = range(1400 * coefficient, 1450 * coefficient)
            cat13 = range(1450 * coefficient, 1500 * coefficient)
            cat14 = range(1500 * coefficient, 1550 * coefficient)
            cat15 = range(1550 * coefficient, 1600 * coefficient)

        categories = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10, cat11, cat13]

        indices = []
        for category in categories:
            indices += list(category)

        return indices

    def save(self, directory=None, file_name=None):
        if directory is None:
            directory = 'reduced'
        file_path = PROJECT_ROOT / 'data' / directory

        if file_name is None and self.is_train:
            file_name = f"Reduced_MVP12_Train_CP.h5"
        elif file_name is None and not self.is_train:
            file_name = f"Reduced_MVP12_Test_CP.h5"

        output_file = h5py.File(file_path / file_name, 'w')

        output_file.create_dataset('incomplete_pcds', data=self.reduced_incomplete_pcds)
        output_file.create_dataset('complete_pcds', data=self.reduced_complete_pcds)
        output_file.create_dataset('labels', data=self.reduced_labels)
        output_file.close()

        print(f"{file_name} was successfully saved at {file_path}.\n")


if __name__ == "__main__":
    mvp_12 = Reduced_MVP12(file_name="MVP_Train_CP.h5", is_train=True)
    # print(mvp_12.labels.shape)
    # print(mvp_12.reduced_incomplete_pcds.shape)
    # print(mvp_12.reduced_labels.shape)
    # print(mvp_12.reduced_complete_pcds.shape)
    mvp_12.save()

    mvp_12 = Reduced_MVP12(file_name="MVP_Test_CP.h5", is_train=False)
    # print(mvp_12.reduced_incomplete_pcds.shape)
    # print(mvp_12.reduced_labels.shape)
    # print(mvp_12.reduced_complete_pcds.shape)

    # for i in range(0, len(mvp_12.reduced_labels), 1300):
    #     print(mvp_12.labels[i:i + 1300])
    #     print("-------------------------------")

    mvp_12.save()
