import numpy as np
import matplotlib.pyplot as plt

# 定义要最大化的函数
def f(x):
    return x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x)

# 初始化参数
N = 50                  # 粒子数量
d = 1                   # 解空间的维数
ger = 500               # 最大迭代次数
limit = [0, 50]         # 位置的边界
vlimit = [-10, 10]      # 速度的边界
w = 0.8                 # 惯性权重
c1 = 0.5                # 认知（自学习）系数
c2 = 0.5                # 社会（群体学习）系数

# 初始化粒子的位置和速度
x = limit[0] + (limit[1] - limit[0]) * np.random.rand(N, d)
v = np.random.rand(N, d)
xm = x.copy()           # 每个粒子的历史最佳位置
ym = np.zeros(d)        # 全局最佳位置
fxm = f(x).copy()       # 每个粒子的历史最佳适应度
fym = np.max(fxm)       # 全局最佳适应度
ym = xm[np.argmax(fxm), :]  # 全局最佳位置

# 绘制初始函数曲线和初始粒子位置
x0 = np.linspace(0, limit[1], 500)
plt.figure()
plt.plot(x0, f(x0), label='函数曲线')
plt.plot(xm, f(xm), 'ro', label='初始粒子位置')
plt.title('初始状态')
plt.legend()
plt.show()

# 优化迭代循环
record = []  # 用于记录每次迭代的最佳适应度

for iter in range(ger):
    fx = f(x)  # 每个粒子的当前适应度

    # 更新每个粒子的历史最佳位置和适应度
    for i in range(N):
        if fx[i] > fxm[i]:
            fxm[i] = fx[i]
            xm[i, :] = x[i, :]

    # 更新全局最佳位置和适应度
    if np.max(fxm) > fym:
        fym = np.max(fxm)
        ym = xm[np.argmax(fxm), :]

    # 更新速度和位置
    v = (w * v + c1 * np.random.rand() * (xm - x)
         + c2 * np.random.rand() * (ym - x))
    v = np.clip(v, vlimit[0], vlimit[1])  # 速度边界处理
    x = x + v
    x = np.clip(x, limit[0], limit[1])    # 位置边界处理

    # 记录当前迭代的最佳适应度，并绘制过程
    record.append(fym)

    plt.subplot(1, 2, 1)
    plt.cla()
    plt.plot(x0, f(x0), 'b-', label='函数曲线')
    plt.plot(x, f(x), 'ro', label='粒子位置')
    plt.title('状态位置变化')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.cla()
    plt.plot(record)
    plt.title('最优适应度进化过程')
    plt.pause(0.01)

plt.show()

# 绘制最终状态图
plt.figure()
plt.plot(x0, f(x0), 'b-', label='函数曲线')
plt.plot(x, f(x), 'ro', label='最终粒子位置')
plt.title('最终状态位置')
plt.legend()
plt.show()

# 输出结果
print(f"最大值：{fym}")
print(f"变量最优取值：{ym}")
