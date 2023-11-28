import torch
import torch.nn.functional as F


input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

print(input.shape)
print(kernel.shape)

#此时不满足conv2d的输入参数，reshape一下，增加通道数和batch
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
output2 = F.conv2d(input, kernel, stride=2)

print("当stride=1的是时候")
print(output1)
print("当stride=2的是时候")
print(output2)

#在input外围加padding+1，也就是填充0
#理论上来说，填充是为了更好地利用边缘特征
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)