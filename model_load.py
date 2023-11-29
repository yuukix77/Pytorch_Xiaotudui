import  torch


#Method1是整个模型保存下来
import torchvision

model = torch.load("vgg16_saveMethod1.pth")
print(model)


#Method2是字典形式只保存参数
vgg16 = torchvision.models.vgg16(pretrained=False)
model2 = vgg16.load_state_dict(torch.load("vgg16_saveMethod2.pth"))
print(model2)

