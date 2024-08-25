import numpy as np
import unittest


# 让函数更易用
def as_array(x):
    if np.isscalar(x):  # 判断是否为np.ndarray实列,否则转成
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):  # 使得Variable只支持ndarray类型
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None  # 创造者

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # 简化y.grad=np.array(1.0)

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

        output = Variable(as_array(y))
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


def square(x):
    f = Square()
    return f(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy

        return gx


def exp(x):
    f = Exp()
    return f(x)


x2 = Variable(np.array(0.5))
y = square(exp(square(x2)))

# 自动反向传播
y.backward()
print(x2.grad)
c = Variable(None)


# a = Variable(1.0)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data + eps)
    x1 = Variable(x.data - eps)
    f0 = f(x0)
    f1 = f(x1)

    return (f0.data - f1.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        y.backward()
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
