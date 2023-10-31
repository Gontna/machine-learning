import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000  # 可以改成1000
learn_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1

train_loader = DataLoader(
    torchvision.datasets.MNIST(root='./data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(
    torchvision.datasets.MNIST(root='./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size=batch_size_test, shuffle=True,
)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # 默认参数是0.5 以p的概率将输出部分设置为0来减少过拟合
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
network = Net()
network.load_state_dict(torch.load("model.pth"))
# print(network)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def test():
    network.eval()
    test_loss =0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            output = network(data)
            test_loss +=F.nll_loss(output,target,size_average=False).item()
            pred = output.data.max(1,keepdim=True)[1]
            correct +=pred.eq(target.data.view_as(pred)).sum() #把target.data转换成和pred一样的形状
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



test()

example = enumerate(test_loader)
batch_idx,(example_data,example_targets) = next(example)
with torch.no_grad():
    output = network(example_data)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
    plt.title("Prediction:{}".format(
        output.data.max(1,keepdim=True)[1][i].item()))
    plt.xticks()
    plt.yticks()
plt.show()
