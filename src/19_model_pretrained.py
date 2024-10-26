import torch
import torchvision

# train_data = torchvision.datasets.ImageNet(root='../data', split='train', download=True,transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

# 在原有的网络层中，新增加一个全连接层
vgg16_true.classifier.add_module('add_linear',torch.nn.Linear(1000,10))
print(vgg16_true)

# 直接修改原有的网络
vgg16_false.classifier[6] = torch.nn.Linear(4096,10)
print(vgg16_false)