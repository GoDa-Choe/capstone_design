import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from auto_encoder import AutoEncoder, AutoEncoderLight
from pointnet import PointNetCls, feature_transform_regularizer

from chamfer_distance import distChamfer
from src.dataset.dataset import MVP
from src.utils.log import get_log_file, logging
from src.utils.weights import get_trained_model_directory, save_trained_model
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
BATCH_SIZE = 32
NUM_CLASSES = 16
FEATURE_TRANSFORM = True

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 4

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")
# RESULT_LOG_ROOT = PROJECT_ROOT / 'result'

PRETRAINED_WEIGHTS = True
PRETRAINED_WEIGHTS_DIRECTORY = PROJECT_ROOT / "pretrained_weights"

GENERATOR_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIRECTORY / "auto_encoder_1024/20211108_101749/170.pth"
DISCRIMINATOR_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIRECTORY / "reduced_complete_reduced_complete/20211108_072210/170.pth"


#####


def evaluate(generator, discriminator, test_loader):
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    category_correct = [0] * 16
    category_count = [0] * 16

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(tqdm(test_loader)):
            if NUM_POINTS != 2024:
                indices = torch.randperm(point_clouds.size()[1])
                indices = indices[:NUM_POINTS]
                point_clouds = point_clouds[:, indices, :]

            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

            point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

            vector, _, _ = generator(point_clouds)

            generated_point_clouds = vector.view(-1, 3, NUM_POINTS)

            scores, trans, trans_feat = discriminator(generated_point_clouds)

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

    test_dataset = MVP(
        is_train=False,
        is_reduced=True,
        shape_type="occluded",
        partition_type='8-axis')

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
        generator.load_state_dict(torch.load(GENERATOR_WEIGHTS_PATH))
        discriminator.load_state_dict(torch.load(DISCRIMINATOR_WEIGHTS_PATH))
    generator.to(device=DEVICE)
    discriminator.to(device=DEVICE)

    log_file = get_log_file("generator", "discriminator")
    # pretrained_weights_directory = get_trained_model_directory(NUM_POINTS)

    test_result = evaluate(generator=generator, discriminator=discriminator, test_loader=test_loader)
    # print(f"{test_result[0] / test_result[1]:.6f}")
    logging(log_file, "test", [0, 0, 1], test_result)

    # log_file.close()
    print(f"The Experiments is ended at {datetime.datetime.now()}.")
