import torch

import os
import datetime

from src.utils.project_root import PROJECT_ROOT


def get_trained_model_directory(dataset_type, train_shape: str):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = PROJECT_ROOT / 'pretrained_weights' / dataset_type / train_shape / now

    try:
        os.makedirs(directory)
    except OSError:
        pass

    return directory


def get_trained_model_directory_for_auto_encoder(dataset_type, loss_type: str):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = PROJECT_ROOT / 'pretrained_weights' / dataset_type / loss_type / now

    try:
        os.makedirs(directory)
    except OSError:
        pass

    return directory


def save_trained_model(model, epoch, directory):
    path = directory / f"{epoch}.pth"
    torch.save(model.state_dict(), path)
