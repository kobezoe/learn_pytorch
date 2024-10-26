import torch
from torch import nn
from torch.ao.nn.qat import Conv2d
from torch.ao.quantization import get_default_qat_qconfig
from torch.nn import MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Hdy(nn.Module):
    def __init__(self):
        super(Hdy, self).__init__()
        # 输入的通道数，输出的通道数量 卷积核
        """
        在初始化时，没有提供以哦个qconfig参数
        在pytorch量化感知训练模块中，conv2d需要一个qconfig参数来量化配置
        """
        qconfig = get_default_qat_qconfig('fbgemm')
        # self.conv1 = Conv2d(3,32,5,padding=2,qconfig=qconfig)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding=2,qconfig=qconfig)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,padding=2,qconfig=qconfig)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64,10)

        self.module1 = Sequential(
            Conv2d(3,32,5,padding=2,qconfig=qconfig),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2,qconfig=qconfig),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2,qconfig=qconfig),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self, x):
        x = self.module1(x)
        return x
hdy = Hdy()
print(hdy)

input = torch.ones((64,3,32,32))
output = hdy(input)
print(output.shape)

writer = SummaryWriter("../logs_seq")
writer.add_graph(hdy,input)
writer.close()