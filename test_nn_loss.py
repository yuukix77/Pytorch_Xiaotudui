import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
resultMSE = loss_mse(inputs, targets)

print("LIloss:{}".format(result))
print("NSEloss:{}".format(resultMSE))

#x为分类结果，y为三分类中的第几类真实target
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])

#（1，3）的 1 是batchsize，3是三分类
x = torch.reshape(x, (1, 3))
print(x.shape)
print(x)
#交叉熵
loss_cross = nn.CrossEntropyLoss()
resultCrossEntropy = loss_cross(x, y)
print("CrossEntropyloss:{}".format(resultCrossEntropy))
