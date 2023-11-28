import  torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5], [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()


    def forward(self, input):
        output = self.relu1(input)
        return  output


class Tudui2(nn.Module):

    def __init__(self):
        super(Tudui2, self).__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

#用测试数据测试
tudui = Tudui()
output = tudui(input)
print(output)

tudui2 = Tudui2()
output = tudui2(input)
print(output)

writer = SummaryWriter("LogsForTestReLUAndSigmoid")
step = 0
for data in dataLoader:
    imgs, targets = data
    writer.add_images("imgsIn1", imgs, step)
    outputImgs1 = tudui(imgs)
    writer.add_images("imgsOut1", outputImgs1, step)
    step = step + 1
print("ReLU处理完成")

step = 0
for data in dataLoader:
    imgs, targets = data
    writer.add_images("imgsIn2", imgs, step)
    outputImgs2 = tudui2(imgs)
    writer.add_images("imgsOut2", outputImgs2, step)
    step = step + 1
print("Sigmoid处理完成")
writer.close()