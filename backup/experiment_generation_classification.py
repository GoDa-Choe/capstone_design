import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from auto_encoder import AutoEncoder, AutoEncoderLight
from pointnet import PointNetCls, feature_transform_regularizer

from chamfer_distance import distChamfer
from src.dataset.dataset import MVP
from src.utils.log import get_log_file, logging
from src.utils.weights_for_auto_encoder import get_trained_model_directory, save_trained_model
from src.visualization.visualize import visualize_generated_and_ground_truth

from tqdm import tqdm
import datetime
from pathlib import Path

# Todo 1. Chamfer Distance Loss Check for train
# Todo 2. Chamfer Distance Loss Check for test
# Todo 3. Regularization required?
# Todo 4. FC Layer Architecture?
# Todo 5. Epoch 100(3.82) -> 200?


#####
NUM_POINTS = 1024
NUM_EPOCH = 200
BATCH_SIZE = 32
NUM_CLASSES = 16
FEATURE_TRANSFORM = True

ALPHA = 1

LEARNING_RATE = 0.01
BETAS = (0.9, 0.999)

STEP_SIZE = 40
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 4

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")
# RESULT_LOG_ROOT = PROJECT_ROOT / 'result'

PRETRAINED_WEIGHTS = True
PRETRAINED_WEIGHTS_DIRECTORY = PROJECT_ROOT / "pretrained_weights"

# GENERATOR_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIRECTORY / "auto_encoder_2048/20211107_052634/99.pth"
DISCRIMINATOR_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIRECTORY / "reduced_complete_reduced_complete/20211108_072210/146.pth"


#####

def train(generator, discriminator, optimizer, lr_schedule, train_loader, pretrained_weights_directory):
    total_loss = 0.0
    total_correct = 0.0
    count = 0

    total_cross_entropy = 0.0
    total_chamfer_distance = 0.0

    generator.train()
    # discriminator.eval()  # Todo 1. check required

    for batch_index, (point_clouds, labels, ground_truths) in enumerate(train_loader):
        if NUM_POINTS != 2024:
            indices = torch.randperm(point_clouds.size()[1])
            indices = indices[:NUM_POINTS]
            point_clouds = point_clouds[:, indices, :]

        point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

        point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

        optimizer.zero_grad()

        # Todo 2. Chamfer Distance Loss
        vector, _, trans_feat1 = generator(point_clouds)
        generated_point_clouds = vector.view(-1, NUM_POINTS, 3)
        dist1, dist2, _, _ = distChamfer(generated_point_clouds, ground_truths)
        cd_loss = ((torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2).mean()
        if FEATURE_TRANSFORM:  # for regularization
            cd_loss += feature_transform_regularizer(trans_feat1) * 0.001
        total_chamfer_distance += cd_loss

        # Todo 2. Cross Entropy Loss
        generated_point_clouds = vector.view(-1, 3, NUM_POINTS)
        scores, _, trans_feat2 = discriminator(generated_point_clouds)
        ce_loss = F.nll_loss(scores, labels)

        if FEATURE_TRANSFORM:  # for regularization
            ce_loss += feature_transform_regularizer(trans_feat2) * 0.001
        total_cross_entropy += ce_loss

        loss = cd_loss * ALPHA * ce_loss
        total_loss += loss
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(scores, 1)
        total_correct += (predictions == labels).sum().item()
        count += labels.size(0)

    if pretrained_weights_directory:
        save_trained_model(generator, epoch, pretrained_weights_directory)

    lr_schedule.step()

    return total_loss, total_correct, count


def evaluate(generator, discriminator, test_loader):
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    total_cross_entropy = 0.0
    total_chamfer_distance = 0.0

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(test_loader):
            if NUM_POINTS != 2024:
                indices = torch.randperm(point_clouds.size()[1])
                indices = indices[:NUM_POINTS]
                point_clouds = point_clouds[:, indices, :]

            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

            point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

            # Todo 2. Chamfer Distance Loss
            vector, _, trans_feat1 = generator(point_clouds)
            generated_point_clouds = vector.view(-1, NUM_POINTS, 3)
            dist1, dist2, _, _ = distChamfer(generated_point_clouds, ground_truths)
            cd_loss = ((torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2).mean()
            if FEATURE_TRANSFORM:  # for regularization
                cd_loss += feature_transform_regularizer(trans_feat1) * 0.001
            total_chamfer_distance += cd_loss

            # Todo 2. Cross Entropy Loss
            generated_point_clouds = vector.view(-1, 3, NUM_POINTS)
            scores, _, trans_feat2 = discriminator(generated_point_clouds)
            ce_loss = F.nll_loss(scores, labels)

            if FEATURE_TRANSFORM:  # for regularization
                ce_loss += feature_transform_regularizer(trans_feat2) * 0.001
            total_cross_entropy += ce_loss

            loss = cd_loss * ALPHA * ce_loss
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

    generator = AutoEncoderLight(num_point=NUM_POINTS, feature_transform=FEATURE_TRANSFORM)
    discriminator = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)

    # for pretrained model
    if PRETRAINED_WEIGHTS:
        # generator.load_state_dict(torch.load(GENERATOR_WEIGHTS_PATH))
        discriminator.load_state_dict(torch.load(DISCRIMINATOR_WEIGHTS_PATH))
    generator.to(device=DEVICE)
    discriminator.to(device=DEVICE)

    for param in discriminator.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    log_file = get_log_file("generator", "discriminator")
    pretrained_weights_directory = get_trained_model_directory(f"{NUM_POINTS}_joint")

    # log_file = None
    # pretrained_weights_directory = None

    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(generator=generator, discriminator=discriminator,
                             optimizer=optimizer, lr_schedule=scheduler,
                             train_loader=train_loader,
                             pretrained_weights_directory=pretrained_weights_directory)
        test_result = evaluate(generator=generator, discriminator=discriminator, test_loader=test_loader)

        logging(log_file, epoch, train_result, test_result)

    # log_file.close()
    print(f"The Experiments is ended at {datetime.datetime.now()}.")
