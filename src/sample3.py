import h5py
import numpy as np

file_path = '../data/MVP_Train_CP.h5'
input_file = h5py.File(file_path, 'r')

print(input_file.attrs.keys())
for attr in input_file.attrs:
    print(attr)
print(input_file['incomplete_pcds'].attrs.keys())
