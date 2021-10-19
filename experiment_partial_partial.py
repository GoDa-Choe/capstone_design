import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from dataset import MVP

import os
import datetime
from pathlib import Path

# Todo 1. scheduler check
# Todo 2. transformation network check
# Todo 3. Saving trained network

#####
NUM_POINTS = 2048
BATCH_SIZE = 32
NUM_CLASSES = 16
NUM_EPOCH = 100
FEATURE_TRANSFORM = True

LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)

STEP_SIZE = 20
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 4

RESULT_LOG_ROOT = 'result/'
TRAINED_MODEL_ROOT = 'trained_model/'
PRETRAINED_MODEL = None

blue = lambda x: '\033[94m' + x + '\033[0m'

#####


partial_train_dataset = MVP(
    shape_type="partial",
    is_train=True,
    root='./data/')

partial_test_dataset = MVP(
    shape_type="partial",
    is_train=False,
    root='./data/')

partial_train_loader = torch.utils.data.DataLoader(
    dataset=partial_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

partial_test_loader = torch.utils.data.DataLoader(
    dataset=partial_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM).to(device=DEVICE)

# @@@for pretrained model@@
# if opt.model != '':
#     classifier.load_state_dict(torch.load(opt.model))
# classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=BETAS)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# num_batch = len(dataset) / opt.batchSize
num_batch = len(partial_test_loader) / BATCH_SIZE


def train(model, lr_schedule, train_loader, trained_model_directory):
    total_loss = 0.0
    total_correct = 0.0
    count = 0

    model.train()

    for batch_index, (point_clouds, labels) in enumerate(train_loader):
        point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)
        point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        scores, trans, trans_feat = model(point_clouds)
        loss = F.nll_loss(scores, labels)

        if FEATURE_TRANSFORM:  # for regularization
            loss += feature_transform_regularizer(trans_feat) * 0.001
        total_loss += loss
        loss.backward()

        optimizer.step()

        _, predictions = torch.max(scores, 1)
        total_correct += (predictions == labels).sum().item()
        count += labels.size(0)

    save_trained_model(model, epoch, trained_model_directory)

    lr_schedule.step()

    return total_loss, total_correct, count


def evaluate(model, test_loader):
    total_loss = 0.0
    total_correct = 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for batch_index, (point_clouds, labels) in enumerate(test_loader):
            point_clouds = point_clouds.transpose(2, 1)  # (batch_size, 2048, 3) -> (batch_size, 3, 2048)
            point_clouds, labels = point_clouds.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            scores, trans, trans_feat = model(point_clouds)
            loss = F.nll_loss(scores, labels)

            if FEATURE_TRANSFORM:  # for regularization
                loss += feature_transform_regularizer(trans_feat) * 0.001
            total_loss += loss

            _, predictions = torch.max(scores, 1)
            total_correct += (predictions == labels).sum().item()
            count += labels.size(0)

    return total_loss, total_correct, count


def logging(file, epoch, train_result, test_result):
    def log_line(loss, correct, count):
        return f"{loss / count:.6f} {correct / count:.6f}"

    train_log = log_line(*train_result)
    test_log = log_line(*test_result)

    print(epoch, train_log, blue(test_log))
    log = f"{epoch} {train_log} {test_log}\n"
    file.write(log)


def get_log_file(train_shape: str, test_shape: str):
    directory = Path(RESULT_LOG_ROOT)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{train_shape}_{test_shape}_{now}.txt"
    file = open(directory / file_name, "w")
    print(
        f"The Experiment from {train_shape.capitalize()} to {test_shape.capitalize()} is started at {datetime.datetime.now()}.")
    index = "Epoch Train_Loss Train_Accuracy Test_Loss Test_Accuracy\n"
    file.write(index)
    print(index, end="")
    return file


def get_trained_model_directory(train_shape: str, test_shape: str):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(TRAINED_MODEL_ROOT) / f"{train_shape}_{test_shape}" / now

    try:
        os.makedirs(directory)
    except OSError:
        pass

    return directory


def save_trained_model(model, epoch, directory):
    from_to, time = directory.parts[-2:]
    path = directory / f"{from_to}_{time}_{epoch}.pth"
    torch.save(model.state_dict(), path)


if __name__ == "__main__":

    file = get_log_file(train_shape="partial", test_shape="partial")
    trained_model_directory = get_trained_model_directory(train_shape="partial", test_shape="partial")

    for epoch in range(NUM_EPOCH):
        train_result = train(model=classifier, lr_schedule=scheduler,
                             train_loader=partial_train_loader, trained_model_directory=trained_model_directory)
        test_result = evaluate(model=classifier, test_loader=partial_test_loader)

        logging(file, epoch, train_result, test_result)

    file.close()
    print(f"The Experiments is ended at {datetime.datetime.now()}.")
