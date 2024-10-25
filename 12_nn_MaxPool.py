import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1,2,0,3,1],
#                      [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]])
#
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataLoader = DataLoader(dataset,batch_size=64)


class Hdy(nn.Module):
    def __init__(self):
        super(Hdy,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output
hdy = Hdy()
writer = SummaryWriter("logs_maxPool")
step = 0
for data in dataLoader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = hdy(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()

# output = hdy(input)
# print(output)

"""
池化层：1080p --->  720p 保留图片的特征
"""