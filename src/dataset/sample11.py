import h5py
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot
from tqdm import tqdm
import random
import time

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")

file_path = PROJECT_ROOT / 'data' / 'partitioned'
file_name = '8-axis_Partitioned_MVP_Train_CP.h5'
# file_name = '8-axis_Partitioned_MVP_Test_CP.h5'
input_file = h5py.File(file_path / file_name, 'r')

print(input_file.keys())
print(input_file['incomplete_pcds'].dtype)
print(input_file['complete_pcds'].dtype)
print(input_file['labels'].dtype)
print(input_file['incomplete_pcds'].shape)
print(input_file['complete_pcds'].shape)
print(input_file['labels'].shape)
