import torch.nn as nn
import torch

# 标准化函数
def conv_bn(inp,oup,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,oup,3,stride,1,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )

#可分离卷积块
def conv_dw(inp,oup,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,oup,3,stride,1,groups=inp,bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        #1*1卷积核调整通道数量
        nn.Conv2d(inp,oup,1,1,0,bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6()
    )


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.stage1 = nn.Sequential(
            #(160,160,3) -> (80,80,32)
            conv_bn(3,32,2),
            conv_dw(32,64,1),


            conv_dw(64,128,2),
            conv_dw(128,128,1),


            conv_dw(128,256,2),
            conv_dw(128,256,1),

        )

        self.stage2= nn.Sequential(
            conv_dw(256,512,2),
            conv_dw(512,512,1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )

        self.stage3 = nn.Sequential(
            conv_dw(512,1024,2),
            conv_dw(1024,1024,1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024,1000)

    def forward(self,x):
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.avg(x)
        x=x.view(-1.1024)
        x= self.fc(x)
        return x

net = MobileNet()
print("12")