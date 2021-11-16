import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.dataset.dataset import Partitioned_MVP
from src.models.pointnet import PointNetCls
from src.utils.log import logging_for_test

from tqdm import tqdm

from src.utils.project_root import PROJECT_ROOT

# Todo 1. scheduler check
# Todo 2. transformation network check
# Todo 3. Saving trained network


#####
NUM_POINTS = 100
BATCH_SIZE = 32
NUM_CLASSES = 16

FEATURE_TRANSFORM = True

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 20


#####


def evaluate(model, test_loader):
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    model.eval()
    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(tqdm(test_loader), start=1):

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
    test_dataset = Partitioned_MVP(
        dataset_type="test",
        pcd_type="occluded")

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)

    WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/partitioned_mvp/occluded/??/35.pth"
    classifier.load_state_dict(torch.load(WEIGHTS_PATH))
    classifier.to(device=DEVICE)

    test_result = evaluate(model=classifier, test_loader=test_loader)

    logging_for_test(test_result)
