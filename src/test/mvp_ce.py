import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.models.auto_encoder import AutoEncoderLight
from src.models.pointnet import PointNetCls

from src.dataset.dataset import MVP

from src.utils.log import logging_for_test

from tqdm import tqdm

from src.utils.project_root import PROJECT_ROOT

#####
INPUT_NUM_POINTS = 100
OUTPUT_NUM_POINTS = 1024
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
            if INPUT_NUM_POINTS != 2024:
                indices = torch.randperm(point_clouds.size()[1])
                indices = indices[:INPUT_NUM_POINTS]
                point_clouds = point_clouds[:, indices, :]

            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
            point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

            vector, trans, trans_feat = generator(point_clouds)
            generated_point_clouds = vector.view(-1, 3, OUTPUT_NUM_POINTS)

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


if __name__ == "__main__":
    test_dataset = MVP(
        dataset_type="test",
        pcd_type="incomplete")

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    generator = AutoEncoderLight(num_point=OUTPUT_NUM_POINTS, feature_transform=FEATURE_TRANSFORM)
    generator.load_state_dict(torch.load(PROJECT_ROOT / "pretrained_weights/mvp/ce/20211117_075102/40.pth"))

    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)
    classifier.load_state_dict(
        torch.load(PROJECT_ROOT / "pretrained_weights/mvp/complete/20211117_044908_for_1024_points/24.pth"))

    generator.to(device=DEVICE)
    classifier.to(device=DEVICE)
    generator.eval()
    classifier.eval()


    test_result = evaluate(generator=generator, classifier=classifier, test_loader=test_loader)
    logging_for_test(test_result)