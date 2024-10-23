from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import  cv2

writer = SummaryWriter("logs")
img = Image.open("image/pytorch.png")
# img = cv2.imread("image/pytorch.png")
# 获取图像的形状
# height, width, channels = img.shape

# print(f'图像的高度: {height}, 宽度: {width}, 通道数: {channels}')
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize 归一化
"""
Normalize a tensor image with mean and standard deviation.
This transform does not support PIL Image.
Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
channels, this transform will normalize each channel of the input
``torch.*Tensor`` i.e.,
``output[channel] = (input[channel] - mean[channel]) / std[channel]``

"""
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5,0.5], [0.5, 0.5, 0.5,0.5 ])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])


# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img resize PIL -> totensor tensor 类型
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize)
print(img_resize)

# compose - > resize -2
trans_resize_2 = transforms.Resize(512)
#  PIL -> PIL-> tensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_size_2 = trans_compose(img)
writer.add_image("Resize_2", img_size_2,1)

#RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_random = trans_compose_2(img)
    writer.add_image("RandomCrop", img_random, i)

"""
    关注输入和输出
    官方文档
    源码中的注释
    关注方法需要什么样的参数
    
    不知道返回值
        print
        print(type())
        debug
"""