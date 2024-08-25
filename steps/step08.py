import numpy as np


# 在反向传播中使用循环而不是使用递归

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None  # 创造者

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            fun = funcs.pop()  # 获取函数
            x, y = fun.input, fun.output  # 获取函数的输入和输出变量
            x.grad = fun.backward(y.grad)  # 调用函数的backward()方法

            if x.creator is not None:  # 如果前面还有函数,则加入到funcs中
                funcs.append(x.creator)


class Function:
    # 在 f = Function() f(x) 时即会调动call函数
    # 输入和输出都是Variable变量
    def __call__(self, input):
        x = input.data
        y = self.forward(x)

        output = Variable(y)
        output.set_creator(self)  # 让输出变量保存创造者信息

        self.input = input  # 保存输入变量
        self.output = output  # 保存输出变量
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

# 尝试反向传播
# y.grad = np.array(1.0)
# C = y.creator  # 获取创造者
# b = C.input  # 获取创造者的输入变量
# b.grad = C.backward(y.grad)  # 调用创造者的backward()
#
# a.grad = B.backward(b.grad)
# x2.grad = A.backward(a.grad)
# print(x2.grad)

# 自动反向传播
y.grad = np.array(1.0)
y.backward()
print(x2.grad)

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x2
