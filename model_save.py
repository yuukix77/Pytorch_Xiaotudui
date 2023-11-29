import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

#保存方式1
#保存模型的结构和模型的参数
torch.save(vgg16, "vgg16_saveMethod1.pth")

#保存方式2
#不保存模型结构，只保存模型参数，以字典的形式（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_saveMethod2.pth")