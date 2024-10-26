class Person:
    def __call__(self,name):
        print("__call__" + "Hello" + name)
    def hello(self,name):
        print("hello" + name)

person = Person()
# 类似java的全参构造
person("zhangsan")
# 调用方法
person.hello("zhangsan01")