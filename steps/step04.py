import numpy as np
from step01_03 import Variable, Square, Exp


# 数值微分 ，使用导数的定义来计算导数，其中h是一个极小值
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data + eps)
    x1 = Variable(x.data - eps)
    f0 = f(x0)
    f1 = f(x1)

    return (f0.data - f1.data) / (2 * eps)


# 求Square类的导数
x = Variable(np.array(2))
f = Square()
dy = numerical_diff(f, x)
print(dy)

# 复合函数求导

x2 = Variable(np.array(0.5))


def f(x2):
    A = Square()
    B = Exp()
    C = Square()

    x2 = C(B(A(x2)))
    return x2


dy = numerical_diff(f, x2)
print(dy)
