

#搭建神经网络
import torch
from torch import nn
from torch.nn.modules.flatten import Flatten


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )
    def forward(self, x):
        x = self.model(x)
        return x

#验证网络正确性
if __name__ == '__main__':
    tudui = Tudui()
    #测试用例的参数分别对应batchsize，channel， 图片的大小H，图片的大小W
    inputTest = torch.ones((64, 3, 32, 32))
    outputTest = tudui(inputTest)
    print(outputTest.shape)