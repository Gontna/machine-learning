import os

import numpy as np
import torch
from torch import nn
import torchvision
import torch.optim as optim

from torchvision import transforms, models, datasets

import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),  # 从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平反转,选择一个概率
        transforms.RandomVerticalFlip(),  # 随机垂直反转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数:亮度 对比度 饱和度 色相
        transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度 3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.465,0.406],[0.229,0.224,0.225])  # 均值,标准差
    ]),
    'valid' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}
data_dir = './fower_data'
train_dir = data_dir+'/train'
valid_dir = data_dir + '/valid'
batch_size = 8
img_datasets = {x : datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train,'valid]}