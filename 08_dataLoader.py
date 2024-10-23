import torchvision

from  torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备测试数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

test_Loader = DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0,drop_last=True)

img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataLoader")
step = 0
for epoch in range(2):
    for data in test_Loader:
        imgs,targets =data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("epoch：{}".format(epoch),imgs,step)
        step = step + 1
writer.close()

"""
    1.batch_size：每次取几个数据，默认是1，一百张牌，一次取几张
    2.shuffle：每一次取数据是按照顺序还是打乱重新取，为true是，打乱重新取，为false是不用打乱
    3.num_workers:是否是多线程？默认是单线程 如果遇到报错，可以调试为0
    4.drop_last:当最后一组数据不能被整除时，为true时，舍去最后一组，为false，不用舍去

"""