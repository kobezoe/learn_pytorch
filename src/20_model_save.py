import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1 模型结构 + 模型参数
torch.save(vgg16,'vgg16_method1.pth')


# 保存方式2 保存模型的参数 官方推荐 占用系统资源较少
torch.save(vgg16.state_dict(),'vgg16_method2.pth')

# 陷阱1
class Hdy(torch.nn.Module):
    def __init__(self):
        super(Hdy, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,6,5)

    def forward(self, x):
        x = self.conv1(x)
        return x
hdy = Hdy()
torch.save(hdy,'hdy.pth')