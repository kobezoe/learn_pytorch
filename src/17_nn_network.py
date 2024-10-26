import torchvision
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

dataset = torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataLoader = DataLoader(dataset,batch_size=64)
class Hdy(nn.Module):
    def __init__(self):
        super(Hdy, self).__init__()
        # 输入的通道数，输出的通道数量 卷积核
        """
        在初始化时，没有提供以哦个qconfig参数
        在pytorch量化感知训练模块中，conv2d需要一个qconfig参数来量化配置
        """
        qconfig = get_default_qat_qconfig('fbgemm')

        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self, x):
        x = self.module1(x)
        return x
hdy = Hdy()
loss = nn.CrossEntropyLoss()
for data in dataLoader:
    input, target = data
    output = hdy(input)
    res_loss = loss(output, target)
    res_loss.backward()
    print(res_loss)