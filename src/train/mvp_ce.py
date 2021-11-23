import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from src.models.pcn import PCN
from src.models.pointnet import PointNetCls

from src.dataset.dataset import MVP
from src.utils.log import get_log_for_auto_encoder, logging_for_train
from src.utils.weights import get_trained_model_directory_for_auto_encoder, save_trained_model

from tqdm import tqdm
import datetime
from src.utils.project_root import PROJECT_ROOT

#####
THRESHOLD = 10

INPUT_NUM_POINTS = 2048
OUTPUT_NUM_POINTS = 2048
BATCH_SIZE = 32
NUM_CLASSES = 16
NUM_EPOCH = 200

FEATURE_TRANSFORM = True

LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)

STEP_SIZE = 20
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 16


#####


def train(generator, classifier, train_loader, lr_schedule):
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cd_loss = 0.0

    total_correct = 0.0
    total_count = 0

    generator.train()

    for batch_index, (point_clouds, labels, ground_truths) in enumerate(train_loader, start=1):
        # if INPUT_NUM_POINTS != 2024:
        #     indices = torch.randperm(point_clouds.size()[1])
        #     indices = indices[:INPUT_NUM_POINTS]
        #     point_clouds = point_clouds[:, indices, :]

        point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        generated_point_clouds = generator(point_clouds)['coarse_output']
        generated_point_clouds = generated_point_clouds.transpose(2, 1)

        scores, _, _ = classifier(generated_point_clouds)

        loss = F.nll_loss(scores, labels)

        total_ce_loss += loss.item()
        loss.backward()

        optimizer.step()

        _, predictions = torch.max(scores, 1)
        total_correct += (predictions == labels).sum().item()
        total_count += labels.size(0)

    lr_schedule.step()

    return total_ce_loss, batch_index, total_correct, total_count


def evaluate(generator, classifier, validation_loader):
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


if __name__ == "__main__":
    train_dataset = MVP(
        dataset_type="train",
        pcd_type="incomplete")

    validation_dataset = MVP(
        dataset_type="validation",
        pcd_type="incomplete")

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

    generator = PCN(emb_dims=1024, input_shape='bnc', num_coarse=2048, detailed_output=False)

    classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)
    WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/mvp/complete/20211117_004344/35.pth"
    classifier.load_state_dict(torch.load(WEIGHTS_PATH))

    generator.to(device=DEVICE)
    classifier.to(device=DEVICE)
    classifier.eval()

    optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    log_file = get_log_for_auto_encoder(dataset_type="mvp", loss_type="ce")
    weights_directory = get_trained_model_directory_for_auto_encoder(dataset_type="mvp", loss_type="ce")

    min_loss = float("inf")
    count = 0
    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(generator=generator, classifier=classifier, train_loader=train_loader,
                             lr_schedule=scheduler)
        validation_result = evaluate(generator=generator, classifier=classifier, validation_loader=validation_loader)

        if validation_result[0] < min_loss:
            save_trained_model(generator, epoch, weights_directory)
            min_loss = validation_result[0]
            count = 0
        else:
            count += 1

        logging_for_train(log_file, epoch, train_result, validation_result)

        if count >= THRESHOLD:
            break

    print(f"The Experiments is ended at {datetime.datetime.now()}.")
