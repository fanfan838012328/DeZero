import numpy as np


# 变量类
class Variable:
    def __init__(self, data):
        self.data = data


# 函数基类
class Function:
    # 在 f = Function() f(x) 时即会调动call函数
    # 输入和输出都是Variable变量
    def __call__(self, input):
        x = input.data
        y = self.forward(x)

        output = Variable(y)

        return output

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


x = Variable(np.array(10))

f = Square()

y = f(x)

print(y.data)
print(type(y))

x2 = Variable(np.array(0.5))
A = Square()
B = Exp()
C = Square()

x2 = A(x2)
x2 = B(x2)
x2 = C(x2)

print(x2.data)
