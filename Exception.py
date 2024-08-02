# 可能会出现错误的代码
# try:
#     x = float("abc")
# # 出现异常的时候，会执行的代码
# except Exception as e:
#     print(e.args)
#     print("can not convert.")
# # 在异常执行之前执行 无论是否异常 都会执行
# finally:
#     x = 0
# # 程序没有结束，会接着执行
# print(x)

def askUser():
    while True:
        try:
            x = float(input("请输入一个float值:"))
            return x
        except ValueError as e:
            print(e.args[0])
            print("请输入一个float值:")


print(askUser())
