import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return  output
#用于测试的5*5tensor
input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]], dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))

#用数据集测试一下
dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64,)

# #用测试tensor测试神经网络
# tudui = Tudui()
# output = tudui(input)
# print(output)

tudui2 = Tudui()
summaryWriter = SummaryWriter("logsForTestMaxPool")
step = 0
#用dataset测试该神经网络
for data in dataloader:
    imgs, targets = data
    outputImages = tudui2(imgs)
    summaryWriter.add_images("imgsIn", imgs, step)
    summaryWriter.add_images("imgsOut", outputImages, step)
    step = step + 1

summaryWriter.close()