import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from urllib3.filepost import writer

dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataLoader = DataLoader(dataset,batch_size=64)

class Hdy(nn.Module):
    def __init__(self):
        super(Hdy, self).__init__()
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

hdy = Hdy()
step = 0
writer = SummaryWriter("logs")
for data in dataLoader:
    imgs,targets =data
    output = hdy(imgs)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_image("output",output,step,dataformats="NCHW")
    step += 1
    print(imgs.shape)
    print(output.shape)
print(hdy)