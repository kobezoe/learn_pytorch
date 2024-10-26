import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset,batch_size=64)
input = torch.tensor([[1,-0.5],
                     [-1,3]])

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

class Hdy(nn.Module):
    def __init__(self):
        super(Hdy,self).__init__()
        self.relu1 = ReLU()
        self.sigmod1 = Sigmoid()

    def forward(self,x):
        # output = self.relu1(x)
        output = self.sigmod1(x)
        return output


step = 0
hdy = Hdy()
writer = SummaryWriter("logs_Relu")
for data in dataLoader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = hdy(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()
