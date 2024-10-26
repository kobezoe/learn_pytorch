import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 默认存储是在cpu上
# ts1 = torch.randn(3,4)
# print(ts1)

# 移动到GPU上
# ts2 = ts1.to('cuda:0')
# print(ts2)


# 展示高清图

# 制作数据集
x1 = torch.rand(10000, 1)  # 输入特征1
x2 = torch.rand(10000, 1)  # 输入特征2
x3 = torch.rand(10000, 1)  # 输入特征3
y1 = ((x1 + x2 + x3) < 1).float()  # 输出特征1
y2 = ((1 < (x1 + x2 + x3)) & ((x1 + x2 + x3) < 2)).float()  # 输出特征2
y3 = ((x1 + x2 + x3) > 2).float()  # 输出特征3
Data = torch.cat([x1, x2, x3, y1, y2, y3], axis=1)
Data.shape
print(Data)
print(len(Data))

# 划分训练集和测试集 以下代码属于通用代码
train_size = int(len(Data) * 0.7)  # 训练集的样本数量
test_size = len(Data) - train_size  # 测试集的样本数量
Data = Data[torch.randperm(Data.size(0)), :]  # 打乱样本的顺序
train_Data = Data[:train_size, :]  # 训练集样本
test_Data = Data[train_size:, :]  # 测试集样本
print(train_Data.shape)
print(test_Data.shape)


# 搭建神经网络
class DNN(nn.Module):
    def __init__(self):
        """搭建神经网络各层"""
        super(DNN, self).__init__()
        self.net = nn.Sequential(  # 按照顺序搭建各层
            nn.Linear(3, 5), nn.ReLU(),  # 第一层 全连接层
            nn.Linear(5, 5), nn.ReLU(),  # 第二层 全连接层
            nn.Linear(5, 5), nn.ReLU(),  # 第三层 全连接层
            nn.Linear(5, 3)  # 第四层 全连接层
        )

    def forward(self, x):
        """前向传播"""
        y = self.net(x)  # x 输入数据
        return y  # y 输出数据


# 创建子类的实例，并且搬到GPU上
# model = DNN().to('cuda:0')
model = DNN()
print(model)

# 查看内部的参数
for name, param in model.named_parameters():
    print(f"参数:{name}\n形状：参数形状:{param.shape}\n数值：{param}\n")

# 激活函数

# 损失函数
loss_fn = nn.MSELoss()

# 学习率和优化算法
learning_rate = 0.01  # 设置学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练网络
epochs = 1000
losses = []  # 记录损失函数变化的列表

# 给训练集划分输入和输出
X = train_Data[:, :3]  # 前三列为输入特征
Y = train_Data[:, :-3]  # 后三列为输出特征

for epoch in range(epochs):
    Pred = model(X)  # 一次前向传播（批量）
    loss = loss_fn(Pred, Y)  # 计算损失函数
    losses.append(loss.item())  # 记录损失函数的变化
    optimizer.zero_grad()  # 清理上一轮滞留的梯度
    loss.backward()  # 一次反向传播
    optimizer.step()  # 优化内部参数
Fig = plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel('loss'), plt.xlabel('epoch')
plt.show()

# 测试网络
X = test_Data[:, :3]
Y = test_Data[:, :-3]

with torch.no_grad():
    Pred = model(X)
    Pred[:, torch.argmax(Pred, axis=1)] = 1
    Pred[Pred != 1] = 0
    correct = torch.sum((Pred == Y).all(1))  # 预测正确的样本
    total = Y.size(0)  # 全部的样本数量
    # todo 为什么测试集的准确度是 0 ？
    print(f'测试集精准度:{100 * correct / total}%')

# 保存网络
torch.save(model, 'model.path')

# 导入网络
new_model = torch.load('model.path')

# 测试网络
# 划分测试集和训练集
X = test_Data[:, :3]
Y = test_Data[:, :-3]

with torch.no_grad():
    Pred = new_model(X)
    Pred[:, torch.argmax(Pred, axis=1)] = 1
    Pred[Pred != 1] = 0
    correct = torch.sum((Pred == Y).all(1))
    total = Y.size(0)
    print(f'测试集精准度:{100 * correct / total}%')
