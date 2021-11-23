import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.models.auto_encoder import AutoEncoderLight, feature_transform_regularizer
from src.models.chamfer_distance import distChamfer
from src.dataset.dataset import Partitioned_MVP

from src.utils.log import get_log_for_auto_encoder, logging_for_cd_train
from src.utils.weights import get_trained_model_directory_for_auto_encoder, save_trained_model

from tqdm import tqdm
import datetime

#####
THRESHOLD = 10

NUM_POINTS = 1024
BATCH_SIZE = 32
NUM_CLASSES = 16
NUM_EPOCH = 200

FEATURE_TRANSFORM = True

LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)

STEP_SIZE = 20
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 20


#####


def train(generator, train_loader, lr_schedule):
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cd_loss = 0.0

    total_correct = 0.0
    total_count = 0

    generator.train()

    for batch_index, (point_clouds, labels, ground_truths) in enumerate(train_loader, start=1):

        point_clouds = point_clouds.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

        optimizer.zero_grad()

        vector, trans, trans_feat = generator(point_clouds)
        generated_point_clouds = vector.view(-1, NUM_POINTS, 3)

        dist1, dist2, _, _ = distChamfer(generated_point_clouds, ground_truths)

        cd_loss = (dist1.mean(1) + dist2.mean(1)).mean()
        if FEATURE_TRANSFORM:  # for regularization
            cd_loss += feature_transform_regularizer(trans_feat) * 0.0001
        total_cd_loss += cd_loss.item()
        cd_loss.backward()

        optimizer.step()

    lr_schedule.step()

    return total_cd_loss, batch_index


def evaluate(generator, validation_loader):
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cd_loss = 0.0

    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    generator.eval()

    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(validation_loader, start=1):
            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
            point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

            vector, trans, trans_feat = generator(point_clouds)
            generated_point_clouds = vector.view(-1, NUM_POINTS, 3)

            dist1, dist2, _, _ = distChamfer(generated_point_clouds, ground_truths)
            cd_loss = (dist1.mean(1) + dist2.mean(1)).mean()

            total_cd_loss += cd_loss

    return total_cd_loss, batch_index


if __name__ == "__main__":
    train_dataset = Partitioned_MVP(
        dataset_type="train",
        pcd_type="occluded")

    validation_dataset = Partitioned_MVP(
        dataset_type="validation",
        pcd_type="occluded")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    generator = AutoEncoderLight(num_point=NUM_POINTS, feature_transform=FEATURE_TRANSFORM)
    generator.to(device=DEVICE)

    optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    log_file = get_log_for_auto_encoder(dataset_type="partitioned_mvp", loss_type="cd")
    weights_directory = get_trained_model_directory_for_auto_encoder(dataset_type="partitioned_mvp", loss_type="cd")

    min_loss = float("inf")
    count = 0
    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(generator=generator, train_loader=train_loader, lr_schedule=scheduler)
        validation_result = evaluate(generator=generator, validation_loader=validation_loader)

        if validation_result[0] < min_loss:
            save_trained_model(generator, epoch, weights_directory)
            min_loss = validation_result[0]
            count = 0
        else:
            count += 1

        logging_for_cd_train(log_file, epoch, train_result, validation_result)

        if count >= THRESHOLD:
            break

    print(f"The Experiments is ended at {datetime.datetime.now()}.")
