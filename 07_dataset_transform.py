import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np


dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform,train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform,train=False,download=True)
# 第三个 参数会将 生成的数据集 变为 tensor

# print(test_set[0])
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("P10")
print("现在没问题")
for i in range(10):
    print("现在仍然没问题")
    img,target = test_set[i]
    print(img)
    print("现在依然没有问题")
    writer.add_image("test_set", img, i)
writer.close()