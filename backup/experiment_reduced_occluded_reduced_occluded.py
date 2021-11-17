import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from pointnet import PointNetCls, feature_transform_regularizer
from src.dataset.dataset import MVP
from src.utils.log import get_log_file, logging
from src.utils.weights import get_trained_model_directory, save_trained_model

from tqdm import tqdm
import datetime
from pathlib import Path

# Todo 1. scheduler check
# Todo 2. transformation network check
# Todo 3. Saving trained network


#####
NUM_POINTS = 100
BATCH_SIZE = 32
NUM_CLASSES = 16
NUM_EPOCH = 100
FEATURE_TRANSFORM = True

LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)

STEP_SIZE = 20
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 4

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")
# RESULT_LOG_ROOT = PROJECT_ROOT / 'result'

ONLY_TEST = False

PRETRAINED_WEIGHTS = None
TRAINED_MODEL_PATH = PROJECT_ROOT / ""


#####


def train(model, lr_schedule, train_loader, pretrained_weights_directory):
    total_loss = 0.0
    total_correct = 0.0
    count = 0

    model.train()

    for batch_index, (point_clouds, labels, ground_truths) in enumerate(train_loader):

        point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

        point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        scores, trans, trans_feat = model(point_clouds)
        loss = F.nll_loss(scores, labels)

        if FEATURE_TRANSFORM:  # for regularization
            loss += feature_transform_regularizer(trans_feat) * 0.001
        total_loss += loss
        loss.backward()

        optimizer.step()

        _, predictions = torch.max(scores, 1)
        total_correct += (predictions == labels).sum().item()
        count += labels.size(0)

    save_trained_model(model, epoch, pretrained_weights_directory)

    lr_schedule.step()

    return total_loss, total_correct, count


def evaluate(model, test_loader):
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    model.eval()
    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(test_loader):
            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

            point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

            scores, trans, trans_feat = model(point_clouds)
            loss = F.nll_loss(scores, labels)

            if FEATURE_TRANSFORM:  # for regularization
                loss += feature_transform_regularizer(trans_feat) * 0.001
            total_loss += loss

            _, predictions = torch.max(scores, 1)
            total_correct += (predictions == labels).sum().item()
            total_count += labels.size(0)

            corrects = (predictions == labels)

            for i in range(len(corrects)):
                label = labels[i]
                category_correct[label] += corrects[i].item()
                category_count[label] += 1

    return total_loss, total_correct, total_count, category_correct, category_count


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

    discriminator = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)

    # for pretrained model
    if PRETRAINED_WEIGHTS:
        discriminator.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    discriminator.to(device=DEVICE)

    optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    log_file = get_log_file(train_shape="reduced_occluded", test_shape="reduced_occluded")
    pretrained_weights_directory = get_trained_model_directory(train_shape="reduced_occluded",
                                                               test_shape="reduced_occluded")

    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(model=discriminator, lr_schedule=scheduler,
                             train_loader=train_loader, pretrained_weights_directory=pretrained_weights_directory)
        test_result = evaluate(model=discriminator, test_loader=test_loader)

        logging(log_file, epoch, train_result, test_result)

    log_file.close()
    print(f"The Experiments is ended at {datetime.datetime.now()}.")
