from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('logs')
# 输入是一个 torch.Tensor, numpy.ndarray, or string/blobname gloabl_step 第一步
img_PIL = Image.open("data/train/bees_image/16838648_415acd9e3f.jpg")
# 转换为 numpy型变量
image_Array = np.array(img_PIL)
# 为啥 最后一个参数不加会报错
# Shape:
#             img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
#             convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
#             Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
#             corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
writer.add_image("test", image_Array, 2,dataformats="HWC")

# y = x
for i in range(100):
    writer.add_scalar("y = x", i, i)


"""
查看tensorboard
    tensorboard --logdir=logs 
避免端口重复
    tensorboard --logdir=logs --port=9999
避免图像都在一个标签中
    杀掉程序 删除原来的logs文件夹下的文件
"""
