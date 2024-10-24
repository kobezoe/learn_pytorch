import torch
import torch.nn.functional as F
from jinja2.nodes import Output
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],[0,1,0],[2,1,0]])


# 输出 tensor.size([5,5]):代表：input是一个张量，一个5行5列的张量
# print(input.shape)

# 变换形状
# 第一维：表示图片的数量，第二维：表示通道数量，第三维：表示图片的宽度，第四维：表示图片的高度
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
# stride ：步长
"""
the stride of the convolving kerenl can be a single number or a tuple(sH,sW) Defaule = 1
"""
output1 = F.conv2d(input,kernel,stride=1)
print(output1)

output2 = F.conv2d(input,kernel,stride=2)
print(output2)

#在input的四周都填充一行（一列）“0“
output3 = F.conv2d(input,kernel,stride=1,padding=1)
print(output3)

# writer = SummaryWriter("logs")
# writer.add_image("input",input,dataformats="NCHW")
# writer.add_image("kernel",kernel,dataformats="NCHW")
# writer.add_image("output",output,dataformats="NCHW")
# writer.close()

