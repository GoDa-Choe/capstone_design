import torch
import torch.nn.parallel
import torch.utils.data

from auto_encoder import AutoEncoder, feature_transform_regularizer
from chamfer_distance import distChamfer
from src.dataset.dataset import MVP
from src.utils.log_for_auto_encoder import get_log_file, logging
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
NUM_POINTS = 2048
BATCH_SIZE = 32
NUM_CLASSES = 16
FEATURE_TRANSFORM = True

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 4

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")
# RESULT_LOG_ROOT = PROJECT_ROOT / 'result'

PRETRAINED_WEIGHTS = True
PRETRAINED_WEIGHTS_DIRECTORY = PROJECT_ROOT / "pretrained_weights"
PRETRAINED_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIRECTORY / "auto_encoder_2048/20211107_052634/99.pth"


#####


def evaluate(model, test_loader):
    total_loss = 0.0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for batch_index, (point_clouds, labels, ground_truths) in enumerate(tqdm(test_loader)):
            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)

            point_clouds, labels, ground_truths = point_clouds.to(DEVICE), labels.to(DEVICE), ground_truths.to(DEVICE)

            vector, trans, trans_feat = model(point_clouds)
            vector = vector.view(-1, NUM_POINTS, 3)

            # for visualization
            # if batch_index % 100 == 0:
            if False:
                generated = vector[0].cpu().detach().numpy()
                ground_truth = ground_truths[0].cpu().detach().numpy()
                label = labels[0].cpu().detach().item()
                visualize_generated_and_ground_truth(generated, ground_truth, label,
                                                     fig_size=(20, 10))

            # Todo 2. Chamfer Distance Loss
            dist1, dist2, _, _ = distChamfer(vector, ground_truths)

            loss = torch.sum(dist1) + torch.sum(dist2)

            if FEATURE_TRANSFORM:  # for regularization
                loss += feature_transform_regularizer(trans_feat) * 0.001
            total_loss += loss
            total_count += labels.size(0)

    return total_loss, total_count


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

    generator = AutoEncoder(num_point=NUM_POINTS, feature_transform=FEATURE_TRANSFORM)

    # for pretrained model
    if PRETRAINED_WEIGHTS:
        generator.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH))
    generator.to(device=DEVICE)

    # log_file = get_log_file(NUM_POINTS)
    # pretrained_weights_directory = get_trained_model_directory(NUM_POINTS)

    test_result = evaluate(model=generator, test_loader=test_loader)
    print(f"{test_result[0] / test_result[1]:.6f}")
    # logging(log_file, epoch, [0, 1], test_result)

    # log_file.close()
    print(f"The Experiments is ended at {datetime.datetime.now()}.")
