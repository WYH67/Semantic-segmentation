# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import math

# ----------------------------------- CrossEntropy loss: base
"""
    class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None)  
    功能：将输入经过 softmax 激活函数之后，再计算其与 target 的交叉熵损失。即该方法将nn.LogSoftmax()和 nn.NLLLoss()进行了结合。严格意义上的交叉熵损失函数应该是nn.NLLLoss()。
    计算公式：loss(x,class) = -log((exp(x[class])) / (∑j exp(x[j]))) = -x[class] + log(∑j exp(x[j]))
    参数：
        weight(Tensor)- 为每个类别的 loss 设置权值，常用于类别不均衡问题。weight 必须是 float类型的 tensor，其长度要于类别 C 一致，即每一个类别都要设置有 weight。带 weight 的计算公式：loss(x,class) = weight[class](-x[class] + log(∑j exp(x[j])))
        size_average(bool)- 当 reduce=True 时有效。为 True 时，返回的 loss 为平均值；为 False时，返回的各样本的 loss 之和。
        reduce(bool)- 返回值是否为标量，默认为 True
        ignore_index(int)- 忽略某一类别，不计算其 loss，其 loss 会为 0，并且，在采用size_average 时，不会计算那一类的 loss，除的时候的分母也不会统计那一类的样本。
        
"""
loss_f = nn.CrossEntropyLoss(weight=None, size_average=True, reduce=False)
# 生成网络输出 以及 目标输出
output = torch.ones(2, 3, requires_grad=True) * 0.5      # 假设一个三分类任务，batchsize=2，假设每个神经元输出都为0.5
target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor) # 把数组转换成类型为long的张量

loss = loss_f(output, target)

print('--------------------------------------------------- CrossEntropy loss: base')
print('loss: ', loss)
print('由于reduce=False，所以可以看到每一个样本的loss，输出为[1.0986, 1.0986]')


# 熟悉计算公式，手动计算第一个样本
output = output[0].detach().numpy()
output_1 = output[0]              # 第一个样本的输出值
target_1 = target[0].numpy()

# 第一项
x_class = output[target_1]
# 第二项
exp = math.e
sigma_exp_x = pow(exp, output[0]) + pow(exp, output[1]) + pow(exp, output[2])
log_sigma_exp_x = math.log(sigma_exp_x)
# 两项相加
loss_1 = -x_class + log_sigma_exp_x
print('---------------------------------------------------  手动计算')
print('第一个样本的loss：', loss_1)


# ----------------------------------- CrossEntropy loss: weight

weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()
loss_f = nn.CrossEntropyLoss(weight=weight, size_average=True, reduce=False) # 各样本的 loss 之和
output = torch.ones(2, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)
loss = loss_f(output, target)
print('\n\n--------------------------------------------------- CrossEntropy loss: weight')
print('loss: ', loss)  #
print('原始loss值为1.0986, 第一个样本是第0类，weight=0.6,所以输出为1.0986*0.6 =', 1.0986*0.6)

# ----------------------------------- CrossEntropy loss: ignore_index

loss_f_1 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=1)
loss_f_2 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=2)

output = torch.ones(3, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
target = torch.from_numpy(np.array([0, 1, 2])).type(torch.LongTensor)

loss_1 = loss_f_1(output, target)
loss_2 = loss_f_2(output, target)

print('\n\n--------------------------------------------------- CrossEntropy loss: ignore_index')
print('ignore_index = 1: ', loss_1)     # 类别为1的样本的loss为0
print('ignore_index = 2: ', loss_2)     # 类别为2的样本的loss为0
