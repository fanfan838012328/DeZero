import numpy as np

from steps.step09_10 import as_array


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


# 修改Function类使其能接受多个变量的输入和输出
class Function:
    # 在 f = Function() f(x) 时即会调动call函数
    # 输入和输出都是Variable变量
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:  # 让输出变量保存创造者信息
            output.set_creator(self)

        self.inputs = inputs  # 保存输入变量
        self.outputs = outputs  # 保存输出变量
        return outputs

    def forward(self, xs):
        raise NotImplementedError

    # 反向传播
    def backward(self, gys):
        raise NotImplementedError


# 实现Add类
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)  # 返回一个元组

    def backward(self, gys):
        return 1


xs = [Variable(np.array(5.0)), Variable(np.array(5.0))]
f = Add()
ys = f(xs)  # ys是元组
y = ys[0]
print(y.data)
