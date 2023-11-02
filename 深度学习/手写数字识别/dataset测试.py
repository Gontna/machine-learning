import torch
from torch.utils.data import DataLoader
import torchvision
dataset =torchvision.datasets.MNIST('./data',download=True,train=True,
                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
dataloader = DataLoader(dataset,shuffle=True)

for item in dataloader:
    print(item)
    # print(y)
    break