import torch
from torch import nn


class Hdy(nn.Module):
    def __init__(self):
       super().__init__()

    def forward(self,input):
        output = input + 1
        return output

hdy = Hdy()
x = torch.tensor(1.0)
output = hdy(x)
print(output)