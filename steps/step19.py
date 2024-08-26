import numpy as np
import weakref
import contextlib


# 扩展Variable类

def as_array(x):
    if np.isscalar(x):  # 判断是否为np.ndarray实列,否则转成
        return np.array(x)
    return x


class Config:
    enable_backprop = True  # 是否启用反向传播


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):  # 使得Variable只支持ndarray类型
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name

        self.grad = None
        self.creator = None  # 创造者
        self.generation = 0  # 辈分

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 输出变量的辈分比创造者大1

    def clear_grad(self):
        self.grad = None

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # 简化y.grad=np.array(1.0)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:  # 使用set确保函数没有重复
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)  # 按照辈分从小到大排序

        add_func(self.creator)
        while funcs:
            fun = funcs.pop()  # 获取函数
            gys = [output().grad for output in fun.outputs]  # 获得输出变量的梯度 , 适应弱引用
            gxs = fun.backward(*gys)  # 计算输入变量的梯度
            if not isinstance(gxs, tuple):  # 可以让输出方便的返回一个元素
                gxs = (gxs,)

            for x, gx in zip(fun.inputs, gxs):  # 将每个输入变量和他们的梯度对应
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:  # 如果前面还有函数,则加入到funcs中
                    add_func(x.creator)

            if not retain_grad:
                for y in fun.outputs:
                    y().grad = None  # 不保留所有输出变量的导数


# 修改Function类使其能接受多个变量的输入和输出
class Function:
    # 在 f = Function() ,f(x) 时即会调动call函数
    # 输入和输出都是Variable变量
    def __call__(self, *inputs):  # 添加*号,接受可变长参数,输入多个参数时,inputs会变成一个列表
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 添加*号解包，如果是列表则输入列表中的每个变量
        if not isinstance(ys, tuple):  # 使得forward函数可以直接输出一个数，不用写成元组
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # 启用反向传播
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])  # 函数的辈份是输入变量的最大辈分

            for output in outputs:  # 让输出变量保存创造者信息
                output.set_creator(self)

            self.inputs = inputs  # 保存输入变量
            self.outputs = [weakref.ref(output) for output in outputs]  # 保存输出变量,对output使用弱引用

        # 如果列表只有一个元素,则返回第一个元素
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    # 反向传播
    def backward(self, gy):
        raise NotImplementedError


# 实现Add类
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y  # 可以只返回一个数

    def backward(self, gy):
        return (gy, gy)


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


def no_grad():
    return using_config("enable_backprop", False)


with using_config("enable_backprop", False):
    x = Variable(np.array(2.0))
    y = square(x)
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x.shape)
print(len(x))
print(x)
