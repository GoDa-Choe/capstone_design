import torch

import os
import datetime
from pathlib import Path

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")


def get_trained_model_directory(train_shape: str, test_shape: str):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = PROJECT_ROOT / 'pretrained_weights' / f"{train_shape}_{test_shape}" / now

    try:
        os.makedirs(directory)
    except OSError:
        pass

    return directory


def save_trained_model(model, epoch, directory):
    path = directory / f"{epoch}.pth"
    torch.save(model.state_dict(), path)
