from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
from urllib3.filepost import writer

# tensor 数据类型
# transform.toTensor
#   1.transform 如何使用
#   2. tensor 数据类型
#  为什么使用相对路径:
#     绝对路径中的转义符 无法被当作字符串使用
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

cv_img = cv2.imread(img_path)
writer = SummaryWriter("logs")
writer.add_image("tensor", tensor_img, 0)
# print(cv_img)

writer.close()