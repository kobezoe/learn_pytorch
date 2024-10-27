import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from trainModel import *

# 准备数据集
train_set = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())


# 查看数据集的长度
train_set_size = len(train_set)
test_set_size = len(test_set)
print("训练集的长度为:{}".format(train_set_size))
print("测试集的长度为:{}".format(test_set_size))

# 加载数据集 利用dataLoader
train_dataLoader = DataLoader(train_set,batch_size=64)
test_dataLoader = DataLoader(test_set,batch_size=64)

# 初始化 模型
hdy = Hdy()
# 总共的训练步骤
total_train_step = 0
# 总共的测试步骤
total_test_step = 0
# 训练的轮数
epoch = 10
# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 优化器 learning_rate = 1e-2
learning_rate = 0.01
optimizer = torch.optim.SGD(hdy.parameters(),lr=learning_rate)
writer = SummaryWriter("../train_model")


for i in range(epoch):
    print("-----第 {} 轮训练开始------".format(i+1))

    # 训练步骤开始
    """
    仅对特定的参数有作用
    """
    hdy.train()
    for data in train_dataLoader:
        imgs,targets = data
        output = hdy(imgs)
        loss = loss_fn(output,targets)

        # 优化器优化模型
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化器
        optimizer.step()
        total_train_step += 1
        # loss.item() 输出的数字本身
        if total_train_step % 100 == 0:
            print("训练次数:{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    total_test_loss = 0
    total_accuracy = 0

    # 测试步骤开始
    hdy.eval()
    with torch.no_grad():
        for data in test_dataLoader:
            imgs,targets = data
            output = hdy(imgs)
            loss = loss_fn(output,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == total_accuracy).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_set_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    # 保存每一轮的模型
    torch.save(hdy,"hdy_{}.pth".format(i))
    print("模型已保存")


writer.close()