import numpy as np
import matplotlib.pyplot as plt

# 定义要最大化的函数
def f(x):
    return x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x)

x = np.linspace(0, 50, 500)  # 从 0 到 50 生成 500 个点

# 计算 y 轴的数据点
y = f(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制函数曲线
plt.plot(x, y, label='f(x) = x * sin(x) * cos(2 * x) - 2 * x * sin(3 * x) + 3 * x * sin(4 * x)')

# 添加标题和标签
plt.title('函数 f(x) 的图像')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()