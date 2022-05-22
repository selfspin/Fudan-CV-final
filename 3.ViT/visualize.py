import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
import numpy as np
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import torchvision
import timm

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="ConvMixer")

parser.add_argument('--batch-size', default=4, type=int)

parser.add_argument('--hdim', default=256, type=int)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    args = parser.parse_args()
    device = torch.device('cuda')

    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train',
               28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle',
               71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose',
               87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree',
               65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake',
               45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark',
               14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal',
               43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange',
               92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo',
               66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver',
               61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray',
               30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider',
               85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower',
               37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout',
               3: 'bear', 58: 'pickup_truck', 16: 'can'}

    normalize = T.Normalize(mean=mean, std=std)
    cifar_test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    cifar_data_path = './data/cifar100'
    cifar_test = CIFAR100(cifar_data_path, train=False, transform=cifar_test_transform)
    cifar_test_loader = DataLoader(cifar_test, batch_size=16, shuffle=True, num_workers=2, pin_memory=False)


    def imshow(img, mean, std, transpose=True):
        std = torch.tensor(std)[:, None, None]
        mean = torch.tensor(mean)[:, None, None]
        npimg = (img * std + mean).numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    dataiter = iter(iter(cifar_test_loader))
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images), mean, std)
    print('GroundTruth:\t', ',\t'.join('%s' % classes[int(x)] for x in labels))

    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
    model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load('model/ViT_small.pth'))
    model.eval()

    outputs = model(F.interpolate(images, size=(224, 224), mode='bilinear'))
    _, predicted = torch.max(outputs, 1)
    print('ViT/S-16 Predicted:\t', ',\t'.join(classes[int(x)] for x in predicted))

    model = torchvision.models.resnet50(pretrained=True)
    num_feature = model.fc.in_features
    model.fc = torch.nn.Linear(num_feature, 100)
    model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load('model/ResNet50.pth'))
    model.eval()

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('ResNet50 Predicted:\t', ',\t'.join(classes[int(x)] for x in predicted))