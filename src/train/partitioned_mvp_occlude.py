import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.dataset.dataset import Partitioned_MVP
from src.models.pointnet import PointNetCls, feature_transform_regularizer
from src.utils.log import get_log_file, logging_for_train
from src.utils.weights import get_trained_model_directory, save_trained_model

from tqdm import tqdm
import datetime

#####
THRESHOLD = 10

NUM_POINTS = 100
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


def train(model, train_loader, lr_schedule):
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    model.train()

    for batch_index, (point_clouds, labels, ground_truths) in enumerate(train_loader, start=1):

        point_clouds = point_clouds.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        scores, trans, trans_feat = model(point_clouds)
        loss = F.nll_loss(scores, labels)

        if FEATURE_TRANSFORM:  # for regularization
            loss += feature_transform_regularizer(trans_feat) * 0.001
        total_loss += loss.item()
        loss.backward()

        optimizer.step()

        _, predictions = torch.max(scores, 1)
        total_correct += (predictions == labels).sum().item()
        total_count += labels.size(0)

    lr_schedule.step()

    return total_loss, batch_index, total_correct, total_count


def evaluate(model, test_loader):
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    model.eval()
    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(test_loader, start=1):

            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
            point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

            scores, trans, trans_feat = model(point_clouds)
            loss = F.nll_loss(scores, labels)

            # if FEATURE_TRANSFORM:  # for regularization
            #     loss += feature_transform_regularizer(trans_feat) * 0.001

            total_loss += loss

            _, predictions = torch.max(scores, 1)
            total_correct += (predictions == labels).sum().item()
            total_count += labels.size(0)

            corrects = (predictions == labels)

            for i in range(len(corrects)):
                label = labels[i]
                category_correct[label] += corrects[i].item()
                category_count[label] += 1

    return total_loss, batch_index, total_correct, total_count, category_correct, category_count


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

    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)
    classifier.to(device=DEVICE)

    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    log_file = get_log_file(experiment_type="train", dataset_type="partitioned_mvp", train_shape="occluded")
    weights_directory = get_trained_model_directory(dataset_type="partitioned_mvp", train_shape="occluded")

    min_loss = float("inf")
    count = 0
    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(model=classifier, train_loader=train_loader, lr_schedule=scheduler)
        validation_result = evaluate(model=classifier, test_loader=validation_loader)

        if validation_result[0] < min_loss:
            save_trained_model(classifier, epoch, weights_directory)
            min_loss = validation_result[0]
            count = 0
        else:
            count += 1

        logging_for_train(log_file, epoch, train_result, validation_result)

        if count >= THRESHOLD:
            break

    print(f"The Experiments is ended at {datetime.datetime.now()}.")
