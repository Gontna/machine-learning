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
torch.manual_seed(random_seed)
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


# 测试数据集内容组成
# batch_id,(example_data,example_target) = next(enumerate(test_loader))
# print(batch_id)
# print(example_target)
# print(example_data.shape)
# print(example_data[1][0])

# flg = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1) #使用子图均匀的将图样放在坐标网络中 args:行,列,第几个图片
#     plt.tight_layout() #是自动调整子图参数，使之填充整个图像区域。 plt.show()函数时会自动运行tight_layout()
#     plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
#     plt.title("Ground Truth:{}".format(example_target[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # 默认参数是0.5 以p的概率将输出部分设置为0来减少过拟合
        self.fc1 = nn.Linear(480, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 480)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learn_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    # model.train()
    # 的作用是启用Batch Normalization 和Dropout。
    # 如果模型中有BN层(Batch
    # Normalization）和Dropout，需要在训练时添加model.train()。model.train()
    # 是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()
    # 是随机取一部分网络连接来训练更新参数。
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度清零
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step() #更新参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network,'./Net.pth') #保存网络结构
            torch.save(network.state_dict(), './model.pth') # 字典保存,保存网络模型的参数到字典里面(占用小,比较推荐)
            torch.save(optimizer.state_dict(), './optimizer.pth')


# train(1)

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


# test()
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()




#------------从保存的模型中继续训练-----------------
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learn_rate, momentum=momentum)

network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

# 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# 不然报错：x and y must be the same size
# 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
for i in range(4, 9):
    test_counter.append(i * len(train_loader.dataset))
    train(i)
    test()

# 刻画训练曲线
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()


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