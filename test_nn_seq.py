import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.modules.flatten import Flatten
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
      def __init__(self):
          super(Tudui, self).__init__()
          #padding = 2 是对着公式算出来的，其他几个参数的设置按照网上给的结构对着抄
          self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
          self.maxpool1 = MaxPool2d(kernel_size=2)
          self.conv2 = Conv2d(in_channels=32, out_channels=32,kernel_size=5, padding=2)
          self.maxpool2 = MaxPool2d(kernel_size=2)
          self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
          self.maxpool3 = MaxPool2d(kernel_size=2)
          self.flatten = Flatten()
          self.linear1 = Linear(1024, 64)
          self.linear2 = Linear(64, 10)
          #将以上换成sequential写法：
          self.model1 = Sequential(
              Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
              MaxPool2d(kernel_size=2),
              Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
              MaxPool2d(kernel_size=2),
              Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
              MaxPool2d(kernel_size=2),
              Flatten(),
              Linear(1024, 64),
              Linear(64, 10),
          )


      def forward(self, x):
          # #冗余写法
          # x = self.conv1(x)
          # x = self.maxpool1(x)
          # x = self.conv2(x)
          # x = self.maxpool2(x)
          # x = self.conv3(x)
          # x = self.maxpool3(x)
          # x = self.flatten(x)
          # x = self.linear1(x)
          # x = self.linear2(x)

          #seq写法
          x = self.model1(x)
          return x

tudui = Tudui()
print(tudui)

#生成测试数据
inputTest = torch.ones(64, 3, 32, 32)
outputTest = tudui(inputTest)
print(outputTest.shape)

#记录训练过程
writer = SummaryWriter("LogsForSeq")
writer.add_graph(tudui, inputTest)
writer.close()

