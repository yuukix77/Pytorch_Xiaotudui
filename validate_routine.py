import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn.modules.flatten import Flatten

#引入一张小狗图片
image_path = "imgs/dog_validate.png"
img = Image.open(image_path)
print(img)

#png图片有四个通道，红黄蓝透明度，所以要用convert只保留RGB三个通道。
#注：如果图片本身只有RGB，也可以这样操作一下，确保"图片格式化"，能在后面适配所有操作，不会出现格式错误
img = img.convert('RGB')

#需要图片是四维的，先弄成3通道，32*32的tensor，处理一下图像
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)
#现在调整为4维
img = torch.reshape(img, (1, 3, 32, 32))
print(img.shape)

# #看一眼图片现在长什么样
# imgPILTransform = torchvision.transforms.ToPILImage()
# imgPIL = imgPILTransform(img)
# imgPIL.show()

#加载模型，要么引入一下wholeTrainModel.py，要么复制过来，否则报错
#因为我们加载的模型就是需要调用tudui = Tudui()
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

model = torch.load("tuduiModelTrainFor30Rounds.pth")
print(model)

#测试模型
#好习惯，加上eval()以及with torch.no_grad()
model.eval()
with torch.no_grad():
    # 不加.cuda()这句报错Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
    # 因为所用的模型输入的图片是imgs = imgs.cuda()类型
    img = img.cuda()
    output = model(img)
print(output)
print(output.argmax(1))

#验证多张
with torch.no_grad():
    # 不加.cuda()这句报错Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
    # 因为所用的模型输入的图片是imgs = imgs.cuda()类型
    img = img.cuda()
    output = model(img)
print(output)
print(output.argmax(1))


#加载数据集看看对没对
dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False,download=True)
print(dataset.classes)
