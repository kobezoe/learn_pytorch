import torch
import torchvision
from torchvision.models import vgg16
# 加载方式 1 ->  保存方式 1
model = torch.load("vgg16_method1.pth")
print(model)

# 加载方式2 ->  保存方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
# 如果不使用 这个方法的话 还是使用 第一种方式加载模型 会造成 加载的都是字典
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

# 陷阱1
"""
这样是不能够加载出保存到模型
    在该文件下，是不能访问到Hdy这个模型，

解决方式：
    1.将模型的代码复制过来
    2.开头加上import 语句
"""
model = torch.load("hdy.pth")
print(model)