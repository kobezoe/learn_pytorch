def foobar(x, y):
    x[1] = 100
    y = 2


# 调用函数 执行函数
# 值传递/引用：作用域问题
x = [1, 2, 3]
y = 1
foobar(x, y)
print(x)
print(y)


# 函数的返回值 不会影响函数的参数原本的值
def fun(z):
    return z ** 2 + 3 * z - 1


z = 2
print(fun(z))


def special_01(x, y):
    # 可以在调用处执行参数的值
    return x - y


# 特殊用户可以在调用处 指定参数的位置和值
print(special_01(y=1, x=2))
x = 1
# x 会去调用默认
# 但是要注意需要把没有默认值的放在前面，不能将没有默认值放在后面
# 引用机制
print(special_01(x, y=2))


# 先定义数组 然后定义一个元组 顺序不能出错
# 参数打包 参数传递
def special_02(x, y, *args, **kwargs):
    print(args)
    print(kwargs)


special_02(1, 2, "hello", 39.8, True, args=1)

"""
first class 一等公民
1.可以被变量进行引用（函数的本质是一个对象）
2.可以作为一个参数传递给其他的函数
3.可以加入到集合中
4.可以作为一个返回值
5.单行函数 Lambda表达式
"""


def f(x):
    return 3 * x ** 2 + 2 * x - 5


def diff(x, func):
    delta = 1e-6
    return (func(x + delta) - func(x)) / delta


#  将函数作为参数传递给另一个函数
print(diff(1, f))

# lambda表达式的写法 lambda 参数列表：函数体
diff(2, lambda x: 3 * x ** 2 + 2 * x - 5)


def f2(x):
    return x ** 2


def f3(x):
    return x ** 3


def f4(x):
    return x ** 4


fun = [f2, f3, f4]
for f in fun:
    print(f(2))


def fun1(x):
    def fun2(y):
        return x + y

    return fun2


fun = fun1(3)
print(fun(1))
# 函数闭包 1.函数中必须内嵌一个函数 2.返回值必须是内嵌函数 3.内嵌函数必须引用外层函数的变量
print(fun1(3)(1))
