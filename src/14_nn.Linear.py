import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataLoader = DataLoader(dataset,batch_size=64)

class Hdy(torch.nn.Module):
    def __init__(self):
        super(Hdy, self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output
hdy = Hdy()
for data in dataLoader:
    imgs,targets  = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    # print(output.shape)
    output = torch.flatten(imgs)
    # print(output.shape)
    output = hdy(output)
    print(output.shape)