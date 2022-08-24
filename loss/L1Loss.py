# coding: utf-8

import torch
import torch.nn as nn

# ----------------------------------- L1 Loss
'''
    class torch.nn.L1Loss(size_average=None, reduce=None)
    功能：计算 output 和 target 之差的绝对值，可选返回同维度的 tensor 或者是一个标量
    计算公式：l(x,y) = L = {l1,...,lN}^T,ln = |xn - yn|
    参数：
        reduce(bool)- 返回值是否为标量，默认为 True
        size_average(bool)- 当 reduce=True 时有效。为 True 时，返回的 loss 为平均值；为 False时，返回的各样本的 loss 之和。

'''
# 生成网络输出 以及 目标输出
'''
    torch.ones(*sizes, out=None) → Tensor    返回一个全为1 的张量，形状由可变参数sizes定义。
    参数：sizes (int...) – 整数序列，定义了输出形状;  out (Tensor, optional) – 结果张量
    示例：torch.ones(2, 3) ——>  1 1 1 
                               1 1 1
'''
output = torch.ones(2, 2, requires_grad=True)*0.5 # requires_grad=True 的作用是让 backward 可以追踪这个参数并且计算它的梯度
target = torch.ones(2, 2)

# 设置三种不同参数的L1Loss
reduce_False = nn.L1Loss(size_average=True, reduce=False) # 返回的各样本的 loss 之和
size_average_True = nn.L1Loss(size_average=True, reduce=True) # 返回的 loss 为平均值
size_average_False = nn.L1Loss(size_average=False, reduce=True) # 返回的 loss 为平均值

o_0 = reduce_False(output, target)
o_1 = size_average_True(output, target)
o_2 = size_average_False(output, target)

print('\nreduce=False, 输出同维度的loss:\n{}\n'.format(o_0))  # tensor([[0.5000, 0.5000],[0.5000, 0.5000]], grad_fn=<L1LossBackward0>)
print('size_average=True，\t求平均:\t{}'.format(o_1)) # 0.5
print('size_average=False，\t求和:\t{}'.format(o_2)) # 0.2
