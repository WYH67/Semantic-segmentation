import torch
import torch.nn as nn
import numpy as np
import math

# NLLLoss实际上是CrossEntropyLoss的最后一步，即对预测结果取负值，然后取log
# 也就是，NLLLoss + logsoftmax = CELoss

m = nn.LogSoftmax(dim=1)  # log(softmax(x))
loss = nn.NLLLoss()

input = torch.randn(3, 5,)
target = torch.randint(high=3,size=(3,))

y=m(input)
output = loss(y, target)
print(output)

