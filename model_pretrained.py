import torchvision.datasets
from torch import nn

vgg16_falsePretrained = torchvision.models.vgg16(pretrained=False)
vgg16_truePretrained = torchvision.models.vgg16(pretrained=True)
print("下载模型完成")
print(vgg16_truePretrained)

#给模型加模块,加再classifier里面,把1000分类再弄成10分类
vgg16_truePretrained.classifier.add_module('add_liner', nn.Linear(1000,10))
print(vgg16_truePretrained)


#更改模型模块，直接将最后一层线性层的的1000分类改为10分类
vgg16_falsePretrained.classifier[6] = nn.Linear(4096, 10)
print(vgg16_falsePretrained)