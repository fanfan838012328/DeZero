import numpy as np


# 反向传播,使用链式法则实现快速求导

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    # 在 f = Function() f(x) 时即会调动call函数
    # 输入和输出都是Variable变量
    def __call__(self, input):
        x = input.data
        y = self.forward(x)

        output = Variable(y)
        # 保存输入变量
        self.input = input

        return output

    def forward(self, x):
        raise NotImplementedError

    # 反向传播
    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy

        return gx


x2 = Variable(np.array(0.5))
A = Square()
B = Exp()
C = Square()

a = A(x2)
b = B(a)
y = C(b)
# 手动反向求导,
# y.grad,x2.grad
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x2.grad = A.backward(a.grad)

print(x2.grad)
