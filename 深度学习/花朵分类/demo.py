import json
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),  # 从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平反转,选择一个概率
        transforms.RandomVerticalFlip(),  # 随机垂直反转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数:亮度 对比度 饱和度 色相
        transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度 3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.465, 0.406], [0.229, 0.224, 0.225])  # 均值,标准差
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = './flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
batch_size = 8
img_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'valid']  # img_dataset=['train','valid']
}

dataloader = {
    x: DataLoader(img_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'valid']
}
data_size = {
    x: len(img_datasets[x])
    for x in ['train', 'valid']
}
class_name = img_datasets['train'].classes

# print(datasets)
# print(dataloader)
# print(data_size)
# print(class_name)

# 读取标签对应的实际名字
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# print(cat_to_name)

#  数据可视化 数据已经被转换成tensor格式,需要重新转为numpy格式还需要返回标准化结果
def img_convert(tensor):
    '''展示数据'''
    img = tensor.to("cpu").clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1, 2, 0)  # 还原成HWC
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # 标准化还原回去
    img = img.clip(0, 1)
    return img


fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2
dataiter = iter(dataloader['valid'])


def myIter(data_list, callback) -> None:
    for item in data_list:
        callback(item)


# for item in dataloader:
#     for iitem, target in dataloader[item]:
#         pass


def oper(item):
    print(item)


myIter(dataloader, oper)

# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
#     ax.set_title(cat_to_name[str(int(class_name[classes[idx]]))])
#     plt.imshow(img_convert(inputs[idx]))
# plt.show()
