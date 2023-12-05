import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn.modules.flatten import Flatten
import os

#----------批量读取图片并做判断--------------
#引入批量图片
#返回该文件夹下所有的文件、文件夹名
print("---------确认图片路径------------")
image_path = []
image_dir = "imgs"
for filename in os.listdir(image_dir):
    image_path.append(image_dir+"/"+filename)
print(image_path)
print("---------确认图片路径------------")

print("---------读取图片并格式化------------")
img = []
for i in range(len(image_path)):
    img.append(Image.open(image_path[i]))
    #png图片有四个通道，红黄蓝透明度，所以要用convert只保留RGB三个通道。
    #注：如果图片本身只有RGB，也可以这样操作一下，确保"图片格式化"，能在后面适配所有操作，不会出现格式错误
    img[i] = img[i].convert('RGB')

    #需要图片是四维的，先弄成3通道，32*32的tensor，处理一下图像
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                               torchvision.transforms.ToTensor()])
    img[i] = transform(img[i])
    print("图片{}的尺寸已更改为：{}".format(i+1, img[0].shape))
    #现在调整为4维
    img[i] = torch.reshape(img[i], (1, 3, 32, 32))
    print("图片{}维度已更改为：{}".format(i+1, img[i].shape))
print("---------读取图片并格式化------------")
# #看一眼图片现在长什么样
# imgPILTransform = torchvision.transforms.ToPILImage()
# imgPIL = imgPILTransform(img)
# imgPIL.show()


print("---------载入model------------")
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
print("---------载入model------------")

#测试模型
#好习惯，加上eval()以及with torch.no_grad()
model.eval()
output = []
with torch.no_grad():
    # 不加.cuda()这句报错Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
    # 因为所用的模型输入的图片是imgs = imgs.cuda()类型
    for i in range(len(img)):
        img[i] = img[i].cuda()
        output.append(model(img[i]))
outputList = []

print("---------分类结果------------")
for i in range(len(output)):
    print("第{}张图片的output为：{}".format(i+1, output[i]))
    outputList.append(output[i].argmax(1).item())
print("所有图片分类集合outputList：{}".format(outputList))
print("---------分类结果------------")
#加载数据集看看对没对
dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False,download=True)
print("分类类别为：")
print(dataset.classes)
classification_objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
answer = []
for i in range(len(img)):
    answer.append(classification_objects[outputList[i]])

print("这{}张图片他们分别为：{}".format(len(img), answer))