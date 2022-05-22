import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import argparse
import os
import random
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='small', type=str)
parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--scale', default=1, type=float)
parser.add_argument('--reprob', default=0.2, type=float)
parser.add_argument('--ra-m', default=12, type=int)
parser.add_argument('--ra-n', default=2, type=int)
parser.add_argument('--jitter', default=0.2, type=float)

parser.add_argument('--wd', default=0.0004, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--workers', default=2, type=int)

parser.add_argument('--device', type=int, nargs='+', default=[2, 3, 4, 6, 7])

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(x) for x in args.device])
seed_value = 0
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
if args.device != 'cpu':
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cifar100_mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
cifar100_std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
    transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std),
    transforms.RandomErasing(p=args.reprob)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])


def test(model, name='ViT'):
    model.eval()
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(tqdm(trainloader, ncols=0)):
        if name[0:3] == 'ViT':
            X = F.interpolate(X, size=(224, 224), mode='bilinear')
        X, y = X.cuda(), y.cuda()

        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    test_loss, test_acc, m = 0, 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(testloader, ncols=0)):
            if name[0:3] == 'ViT':
                X = F.interpolate(X, size=(224, 224), mode='bilinear')
            X, y = X.cuda(), y.cuda()

            with torch.cuda.amp.autocast():
                output = model(X)
            loss = criterion(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    print(
        f'[{name}] | Train Acc: {train_acc / n:.4f}, '
        f'Test Acc: {test_acc / m:.4f}, Time: {time.time() - start:.1f}')


criterion = nn.CrossEntropyLoss()

trainset = torchvision.datasets.CIFAR100(root='data/cifar100', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR100(root='data/cifar100', download=True, train=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
model = nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load('model/ViT_small.pth'))
test(model, 'ViT/S-16')

model = torchvision.models.resnet50(pretrained=True)
num_feature = model.fc.in_features
model.fc = torch.nn.Linear(num_feature, 100)
model = nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load('model/ResNet50.pth'))
test(model, 'ResNet50')
