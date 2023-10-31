import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           download=True,
                                           train=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,)
                                               )]
                                           ))
train_loader = DataLoader(train_dataset)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,5,5)

    def forward(self,x):
        x=