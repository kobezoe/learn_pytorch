import numpy as np

# 创建一维数组 向量
a = np.array([1, 2, 3, 4, 5])
print(a)

# 创建二维数组 矩阵
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

# 创建二维数组 行矩阵
c = np.array([[1, 2, 3]])
print(c)

# 创建二维数组 列矩阵
d = np.array([[1], [2], [3]])
print(d)

# 创建递增数组 - 从 0 开始 到10 结束
e = np.arange(10)
print(e)

# 创建递增数组 - 从 10 开始 到 20 之前结束
f = np.arange(10, 20)
print(f)

# 创建递增数组 - 从1 开始 到21结束  步长为 2
g = np.arange(1, 21, 2)
print(g)

# 创建全 0 数组 形状为3
h = np.zeros(3)
print(h)

# 创建全1数组 形状为(1,3)的矩阵
i = np.ones((1, 3))

# 创建全 3.14的数组
j = 3.14 * np.ones((3, 3))

# 创建随机函数 0-1 均匀分布的浮点型随机数组
k = np.random.random(5)

l = np.random.randint(1, 10, 5)

# 服从正态分布的随机数组
m = np.random.normal(1, 10, (3, 3))

# 一维数组（向量）的转置
arr1 = np.arange(1, 4)

arr2 = arr1.reshape((1, -1))

arr3 = arr2.T
print(arr3)

# 二维数组(矩阵)的转置
arr4 = np.arange(4).reshape(2, 2)
print(arr4)
arr5 = arr4.T
print(arr5)
