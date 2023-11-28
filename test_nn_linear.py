import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision import transforms

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)
    def forward(self, input):
        out = self.linear1(input)
        return out


dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    input = torch.flatten(imgs)
    output = tudui(input)
    print(output.shape)