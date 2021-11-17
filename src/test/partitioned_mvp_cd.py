import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.models.auto_encoder import AutoEncoderLight
from src.models.pointnet import PointNetCls
from src.dataset.dataset import Partitioned_MVP
from src.models.chamfer_distance import distChamfer

from src.utils.log import logging_for_test, logging_for_cd_test

from tqdm import tqdm
from src.utils.project_root import PROJECT_ROOT

#####
NUM_POINTS = 1024
BATCH_SIZE = 32
NUM_CLASSES = 16

FEATURE_TRANSFORM = True

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 20


#####


def evaluate(generator, classifier, test_loader):
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cd_loss = 0.0

    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(tqdm(test_loader), start=1):

            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
            point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

            vector, _, _ = generator(point_clouds)
            generated_point_clouds = vector.view(-1, 3, NUM_POINTS)

            scores, _, _ = classifier(generated_point_clouds)
            loss = F.nll_loss(scores, labels)

            # if FEATURE_TRANSFORM:  # for regularization
            #     loss += feature_transform_regularizer(trans_feat) * 0.001

            total_ce_loss += loss

            _, predictions = torch.max(scores, 1)
            total_correct += (predictions == labels).sum().item()
            total_count += labels.size(0)

            corrects = (predictions == labels)

            for i in range(len(corrects)):
                label = labels[i]
                category_correct[label] += corrects[i].item()
                category_count[label] += 1

    return total_ce_loss, batch_index, total_correct, total_count, category_correct, category_count


def evaluate_for_cd(generator, validation_loader):
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cd_loss = 0.0

    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(tqdm(validation_loader), start=1):
            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
            point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

            vector, _, _ = generator(point_clouds)
            generated_point_clouds = vector.view(-1, NUM_POINTS, 3)

            dist1, dist2, _, _ = distChamfer(generated_point_clouds, ground_truths)
            cd_loss = (dist1.mean(1) + dist2.mean(1)).mean()

            total_cd_loss += cd_loss

    return total_cd_loss, batch_index


if __name__ == "__main__":
    test_dataset = Partitioned_MVP(
        dataset_type="test",
        pcd_type="occluded")

    validation_dataset = Partitioned_MVP(
        dataset_type="validation",
        pcd_type="occluded"
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    generator = AutoEncoderLight(num_point=NUM_POINTS, feature_transform=FEATURE_TRANSFORM)
    generator.load_state_dict(torch.load(PROJECT_ROOT / "pretrained_weights/partitioned_mvp/cd/20211117_130134/24.pth"))

    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)
    classifier.load_state_dict(
        torch.load(PROJECT_ROOT / "pretrained_weights/mvp/complete/20211117_044908_for_1024_points/24.pth"))

    generator.to(device=DEVICE)
    classifier.to(device=DEVICE)

    generator.eval()
    classifier.eval()

    test_result = evaluate(generator=generator, classifier=classifier, test_loader=validation_loader)
    logging_for_test(test_result)

    validation_result = evaluate_for_cd(generator=generator, validation_loader=validation_loader)
    logging_for_cd_test(validation_result)
