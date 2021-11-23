import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.models.auto_encoder import AutoEncoderLight
from src.models.pcn import PCN
from src.models.pointnet import PointNetCls
from src.dataset.dataset import MVP
from src.models.chamfer_distance import distChamfer

from src.utils.log import logging_for_test, logging_for_cd_test

from tqdm import tqdm
from src.utils.project_root import PROJECT_ROOT

#####
INPUT_NUM_POINTS = 2048
OUTPUT_NUM_POINTS = 2048
BATCH_SIZE = 32
NUM_CLASSES = 16

FEATURE_TRANSFORM = True

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 16


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
            # if INPUT_NUM_POINTS != 2024:
            #     indices = torch.randperm(point_clouds.size()[1])
            #     indices = indices[:INPUT_NUM_POINTS]
            #     point_clouds = point_clouds[:, indices, :]

            point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

            generated_point_clouds = generator(point_clouds)['coarse_output']
            generated_point_clouds = generated_point_clouds.transpose(2, 1)

            scores, _, _ = classifier(generated_point_clouds)
            loss = F.nll_loss(scores, labels)

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
            # if INPUT_NUM_POINTS != 2024:
            #     indices = torch.randperm(point_clouds.size()[1])
            #     indices = indices[:INPUT_NUM_POINTS]
            #     point_clouds = point_clouds[:, indices, :]

            point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

            generated_point_clouds = generator(point_clouds)['coarse_output']

            dist1, dist2, _, _ = distChamfer(generated_point_clouds, ground_truths)
            cd_loss = (dist1.mean(1) + dist2.mean(1)).mean()

            total_cd_loss += cd_loss

    return total_cd_loss, batch_index


if __name__ == "__main__":
    test_dataset = MVP(
        dataset_type="test",
        pcd_type="incomplete")

    validation_dataset = MVP(
        dataset_type="validation",
        pcd_type="incomplete"
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

    generator = PCN(emb_dims=1024, input_shape='bnc', num_coarse=2048, grid_size=4, detailed_output=False)
    generator.load_state_dict(torch.load(PROJECT_ROOT / "pretrained_weights/mvp/cd/20211123_135758/8.pth"))

    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)
    classifier.load_state_dict(
        torch.load(PROJECT_ROOT / "pretrained_weights/mvp/complete/20211117_004344/35.pth"))

    generator.to(device=DEVICE)
    classifier.to(device=DEVICE)

    generator.eval()
    classifier.eval()

    # test_result = evaluate(generator=generator, classifier=classifier, test_loader=test_loader)
    # logging_for_test(test_result)

    test_result = evaluate(generator=generator, classifier=classifier, test_loader=validation_loader)
    logging_for_test(test_result)

    # validation_result = evaluate_for_cd(generator=generator, validation_loader=validation_loader)
    # logging_for_cd_test(validation_result)
