import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from auto_encoder import AutoEncoder, AutoEncoderLight, feature_transform_regularizer
from chamfer_distance import distChamfer
from src.dataset.dataset import MVP
from src.utils.log_for_auto_encoder import get_log_file, logging
from src.utils.weights_for_auto_encoder import get_trained_model_directory, save_trained_model

from tqdm import tqdm
import datetime
from pathlib import Path

# Todo 1. Chamfer Distance Loss Check for train
# Todo 2. Chamfer Distance Loss Check for test
# Todo 3. Regularization required?
# Todo 3. FC Layer Architecture?


#####
NUM_POINTS = 1024
BATCH_SIZE = 32
NUM_CLASSES = 16
NUM_EPOCH = 300
FEATURE_TRANSFORM = True

LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)

STEP_SIZE = 40
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 4

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")
# RESULT_LOG_ROOT = PROJECT_ROOT / 'result'

ONLY_TEST = False

TRAINED_MODEL_PATH = PROJECT_ROOT / ""

PRETRAINED_WEIGHTS = False
PRETRAINED_WEIGHTS_DIRECTORY = PROJECT_ROOT / "pretrained_weights"

AUTO_ENCODER_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIRECTORY / "auto_encoder_2048/20211107_052634/99.pth"


#####


def train(model, lr_schedule, train_loader, pretrained_weights_directory):
    total_loss = 0.0
    count = 0

    model.train()

    for batch_index, (point_clouds, labels, ground_truths) in enumerate(train_loader):

        # sampling
        if NUM_POINTS != 2024:
            indices = torch.randperm(point_clouds.size()[1])
            indices = indices[:NUM_POINTS]
            point_clouds = point_clouds[:, indices, :]

        point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

        point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

        optimizer.zero_grad()

        vector, trans, trans_feat = model(point_clouds)
        vector = vector.view(-1, NUM_POINTS, 3)

        # Todo 1. Chamfer Distance Loss
        dist1, dist2, _, _ = distChamfer(vector, ground_truths)
        loss = ((torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2).mean()

        if FEATURE_TRANSFORM:  # for regularization
            loss += feature_transform_regularizer(trans_feat) * 0.001
        total_loss += loss
        loss.backward()

        optimizer.step()

        count += labels.size(0)

    save_trained_model(model, epoch, pretrained_weights_directory)

    lr_schedule.step()

    return total_loss, count


def evaluate(model, test_loader):
    total_loss = 0.0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(test_loader):
            # sampling
            if NUM_POINTS != 2024:
                indices = torch.randperm(point_clouds.size()[1])
                indices = indices[:NUM_POINTS]
                point_clouds = point_clouds[:, indices, :]

            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

            point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

            vector, trans, trans_feat = model(point_clouds)
            vector = vector.view(-1, NUM_POINTS, 3)

            # Todo 2. Chamfer Distance Loss
            dist1, dist2, _, _ = distChamfer(vector, ground_truths)
            loss = ((torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2).mean()

            if FEATURE_TRANSFORM:  # for regularization
                loss += feature_transform_regularizer(trans_feat) * 0.001
            total_loss += loss
            total_count += labels.size(0)

    return total_loss, total_count


if __name__ == "__main__":
    train_dataset = MVP(
        is_train=True,
        is_reduced=True,
        shape_type="occluded",
        partition_type='8-axis')

    test_dataset = MVP(
        is_train=False,
        is_reduced=True,
        shape_type="occluded",
        partition_type='8-axis')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    generator = AutoEncoderLight(num_point=NUM_POINTS, feature_transform=FEATURE_TRANSFORM)

    # for pretrained model
    if PRETRAINED_WEIGHTS:
        generator.load_state_dict(torch.load(AUTO_ENCODER_WEIGHTS_PATH))
    generator.to(device=DEVICE)

    optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    log_file = get_log_file(NUM_POINTS)
    pretrained_weights_directory = get_trained_model_directory(NUM_POINTS)

    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(model=generator, lr_schedule=scheduler,
                             train_loader=train_loader, pretrained_weights_directory=pretrained_weights_directory)
        test_result = evaluate(model=generator, test_loader=test_loader)

        logging(log_file, epoch, train_result, test_result)

    log_file.close()
    print(f"The Experiments is ended at {datetime.datetime.now()}.")
