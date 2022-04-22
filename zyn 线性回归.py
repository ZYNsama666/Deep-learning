import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
'''
线性回归的重点在于损失函数的求导
计算损失函数非常简单，只要现在的函数生成的y与目标y之间差值绝对值即可
但是为了能够利用torch的autograd，不能把tensor拆开了去算
得整个处理，所以在这里使用tensor相减再pow的方式规避abs的使用
tensor的pow不是矩阵平方，而是矩阵中每个标量的平方
自动求导过程：
 1、声明要求导的函数时加requires_grad=True
 2、在运算时保证loss对要求导函数的运算是连续的
 3、注意loss求导要带负号，所以应该是减回去不是加回去
 4、每次求导完一定是数值（data）加回去，不用原矩阵，防止不能zero
 5、全部运算完后要zero保证梯度不积累
'''
theta = torch.randn(1, 2, requires_grad=True)  # 参数矩阵，分别记录y=ax+b中的a和b
size = 5  # 待拟合集数量
x = torch.Tensor([[1.4, 5, 11, 16, 21]])  # x值
y = torch.Tensor([[14.4, 29.6, 62, 85.5, 113.4]])  # y值
Training_size = 1000  # 训练次数
alpha = 1e-5   # 学习率
output = torch.zeros(1, 5)  # 记录过程中的ab生成的值


def training():
    global theta, output
    new_x = torch.cat((x, torch.ones(1, 5)), 0)
    output = torch.mm(theta, new_x)
    loss = (torch.mm(theta, new_x) - y).pow(2).sum()
    loss.backward()
    theta.data -= theta.grad * alpha
    theta.grad.zero_()
    return loss


for i in range(Training_size):
    _loss = training()
    if i % 50 == 0:
        plt.cla()  # 绘图部分
        plt.scatter(x.numpy(), y.numpy())
        plt.plot(x.numpy()[0], output.data.numpy()[0], 'r-', lw=5)
        plt.text(0.5, 0, 'loss=%s' % (_loss.item()), fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.5)
