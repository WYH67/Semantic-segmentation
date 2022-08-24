# coding: utf-8

import torch
import torch.nn as nn

# ----------------------------------- MSE loss
'''
    class torch.nn.MSELoss(size_average=None, reduce=None, reduction='elementwise_mean')
    功能：计算 output 和 target 之差的平方，可选返回同维度的 tensor 或者是一个标量。
    计算公式：l(x,y) = L{l1,...,lN}^T,ln=(xn - yn)^2
    参数：
        reduce(bool)- 返回值是否为标量，默认为 True
        size_average(bool)- 当 reduce=True 时有效。为 True 时，返回的 loss 为平均值；为 False时，返回的各样本的 loss 之和。
'''

# 生成网络输出 以及 目标输出
output = torch.ones(2, 2, requires_grad=True) * 0.5
target = torch.ones(2, 2)

# 设置三种不同参数的L1Loss
reduce_False = nn.MSELoss(size_average=True, reduce=False)
size_average_True = nn.MSELoss(size_average=True, reduce=True)
size_average_False = nn.MSELoss(size_average=False, reduce=True)


o_0 = reduce_False(output, target)
o_1 = size_average_True(output, target)
o_2 = size_average_False(output, target)

print('\nreduce=False, 输出同维度的loss:\n{}\n'.format(o_0))  # tensor([[0.2500, 0.2500],[0.2500, 0.2500]], grad_fn=<MseLossBackward0>)
print('size_average=True，\t求平均:\t{}'.format(o_1)) # 0.25
print('size_average=False，\t求和:\t{}'.format(o_2)) # 1.0
