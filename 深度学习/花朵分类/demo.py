import json
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# 数据预处理
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
# def img_convert(tensor):
#     '''展示数据'''
#     img = tensor.to("cpu").clone().detach()
#     img = img.numpy().squeeze()
#     img = img.transpose(1, 2, 0)  # 还原成HWC
#     img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # 标准化还原回去
#     img = img.clip(0, 1)
#     return img
#
#
# fig = plt.figure(figsize=(20, 12))
# columns = 4
# rows = 2
# dataiter = iter(dataloader['valid'])
# inputs, classes = next(dataiter)
#
# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
#     ax.set_title(cat_to_name[str(int(class_name[classes[idx]]))])
#     plt.imshow(img_convert(inputs[idx]))
# plt.show()


# 是否用GPU训练
train_on_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义冻结函数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#  迁移学习
model_name = 'resnet'
feature_extract = True  # 是否用已经训练好的特征


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

#  计算设备
model_ft = model_ft.to(device)
# 模型保存
filename = 'checkpoint.pth'
# 是否训练所有层
params_to_update = model_ft.parameters()
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

#   定义优化器
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
# 学习率每7个epoch衰减成原来的1/10
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()
