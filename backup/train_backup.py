import torch.optim as optim
import torch.utils.data
# from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from src.models.pointnet import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from src.dataset.dataset import MVP

#####

BATCH_SIZE = 32
NUM_CLASSES = 16
NUM_EPOCH = 250
FEATURE_TRANSFORM = True

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 1
PRE_TRAINED_MODEL_ROOT = 'trained_model/'

blue = lambda x: '\033[94m' + x + '\033[0m'

#####
""" 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='trained_model folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')
"""

complete_train_dataset = MVP(
    shape_type="complete",
    is_train=True,
    root='./data/')

complete_test_dataset = MVP(
    shape_type="complete",
    is_train=False,
    root='./data/')

train_loader = torch.utils.data.DataLoader(
    dataset=complete_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=NUM_WORKERS
)

test_loader = torch.utils.data.DataLoader(
    dataset=complete_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    # num_workers=NUM_WORKERS
)

# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=opt.batchSize,
#     shuffle=True,
#     num_workers=int(opt.workers))

# testdataloader = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=opt.batchSize,
#     shuffle=True,
#     num_workers=int(opt.workers))

# print(len(dataset), len(test_dataset))
# num_classes = len(dataset.classes)
# print('classes', num_classes)

# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass

# classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM)
classifier = PointNetCls(k=NUM_CLASSES, feature_transform=FEATURE_TRANSFORM).to(device=DEVICE)

# @@@for pretrained model@@
# if opt.model != '':
#     classifier.load_state_dict(torch.load(opt.model))
# classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# num_batch = len(dataset) / opt.batchSize
num_batch = len(complete_train_dataset) / BATCH_SIZE

for epoch in range(NUM_EPOCH):
    # scheduler.step()
    current_loss = 0.0
    correct = 0
    count = 0

    for i, data in enumerate(train_loader, 0):
        points, target = data
        # target = target[:, 0]
        points = points.transpose(2, 1)
        # points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        # print("pred", pred.shape)
        # print("target", target.shape, target)
        loss = F.nll_loss(pred, target)
        if FEATURE_TRANSFORM:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        # # for monitoring
        # values, predictions = torch.max(pred, 1)
        # print("predictions", predictions)
        # print("target", target)
        # correct += (predictions == target).sum().item()  # accumulate correct trained_model per mini-batch
        # count += target.size(0)
        #
        # current_loss += loss.item()
        # if i % 20 == (20 - 1):
        #     current = i * len(points)
        #     total = len(train_loader.dataset)
        #     progress_rate = i / len(train_loader) * 100
        #
        #     log = f"""Train Epoch:{epoch} [{current}/{total}({progress_rate:.1f}%)] """ \
        #           f"""Train Loss: {current_loss / 20:.6f}"""
        #     print(log)
        #     current_loss = 0.0
        #
        # log = f"""
        #             EPOCH: {epoch}    Train Accuracy: {correct / count * 100:.2f}%
        #             """
        # print(log)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(BATCH_SIZE)))

        # if i % 10 == 0:
        #     j, data = next(enumerate(test_loader, 0))
        #     points, target = data
        #     # target = target[:, 0]
        #     points = points.transpose(2, 1)
        #     # points, target = points.cuda(), target.cuda()
        #     classifier = classifier.eval()
        #     pred, _, _ = classifier(points)
        #     loss = F.nll_loss(pred, target)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(target.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
        #         epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(BATCH_SIZE)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (PRE_TRAINED_MODEL_ROOT, epoch))
    scheduler.step()

total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(test_loader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
