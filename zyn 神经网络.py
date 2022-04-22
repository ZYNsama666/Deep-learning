import numpy as np
import math
import matplotlib.pyplot as plt
'''
神经网络利用的就是著名的黑盒原理
我塞进去一些数，神经网络通过一定的加权计算等进行我希望他做的事情
当然一开始他肯定是做不好的
我们并不需要知道他怎么去做好
我们只需要不停的修改他的参数，让他慢慢的符合我们的需求即可。

一般而言，神经网络都是以概率的形式进行判断的，比如机器对于一张猫的照片打分，一共有猫，车，青蛙三种打分类
机器给这张照片是猫的打分为1，是车的打分为4，是青蛙的打分是-2，就说明机器认为这张照片是车的概率最大
现在这个机器猜错了
我们需要不断的训练这个机器
让这个机器有更大的概率给照片对应的事物打更高的分

这里只是一个基础的神经网络，目的是把输入的input1和input2转换为expect_o1和expect_o2

'''

_w = np.zeros(9)       # 黑盒参数使用了1-8,0空出来
i1 = 0.16              # 输入1 不可改
i2 = 0.1               # 输入2 不可改
expect_o1 = 0.01       # 期待输出1
expect_o2 = 0.99       # 期待输出2
b1 = 0                 # 参数b1 不可改
b2 = 0                 # 参数b2 不可改
alpha = 0.1            # 学习率
_plotLOSS = []         # Y轴 损失函数
_plotTIME = []         # X轴 时间
MAX_round = 10000      # 训练次数


def My_sigmoid(x):    # 手写sigmoid函数
    ans = 1 / (1 + np.exp(-x))
    return ans


def math_ans(_input1, _input2, _w1, _w2, _b):    # 计算net
    _net = _w1 * _input1 + _w2 * _input2 + _b
    return My_sigmoid(_net)


def math_update(_W, _o, expect_o, _h):    # 更新权重
    _derivativeE = -(expect_o - _o)
    _derivativeO = _o * (1 - _o)
    _derivativeN = _h
    _derivative = _derivativeE * _derivativeO * _derivativeN
    new_W = _W - alpha * _derivative
    return new_W


def training(_round):    # 一次训练
    _h1 = math_ans(i1, i2, _w[1], _w[2], b1)
    _h2 = math_ans(i1, i2, _w[3], _w[4], b1)
    _o1 = math_ans(_h1, _h2, _w[5], _w[6], b2)
    _o2 = math_ans(_h1, _h2, _w[7], _w[8], b2)
    _loss = 0.5 * (_o1 - expect_o1) ** 2 + 0.5 * (_o2 - expect_o2) ** 2
    _w[5] = math_update(_w[5], _o1, expect_o1, _h1)
    _w[6] = math_update(_w[6], _o1, expect_o1, _h2)
    _w[7] = math_update(_w[7], _o2, expect_o2, _h1)
    _w[8] = math_update(_w[8], _o2, expect_o2, _h2)
    _w[1] = math_update(_w[1], _h1, expect_o1, i1)
    _w[2] = math_update(_w[2], _h1, expect_o1, i2)
    _w[3] = math_update(_w[3], _h2, expect_o2, i1)
    _w[4] = math_update(_w[4], _h2, expect_o2, i2)
    if _round % 500 == 0:
        _plotLOSS.append(_loss)
        _plotTIME.append(_round)
        print("round:" + str(_round) + " loss=" + str(_loss))
    if _round == MAX_round-1:
        print("o1=" + str(_o1) + " o2=" + str(_o2))


def main():
    print("training set:")
    for i in range(MAX_round):
        training(i)
    plt.figure(1)
    plt.plot(_plotTIME, _plotLOSS, label='train_loss')
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()


main()
