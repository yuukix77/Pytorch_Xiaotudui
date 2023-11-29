import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader
import time
#---------------一种更简洁的调用GPU训练的方式------------------
#使用torch.device()定义训练的设备，然后使用to(XXX)将变量放在device上训练
#注意，to这个方法，在model、loss中是不用重复赋值的，在imgs、targets上要，为了统一和方便记忆，都赋值是好习惯
#---------------只有网络模型、loss、imgs、targets有cuda方法-----------------

#定义一个设备
device0 = torch.device("cuda")
#device1 = torch.device("cpu")
#device2 =torch.device("cuda:0") #如果有多个GPU就可以用:X指定用哪个GPU训练
print("现在使用的设备是：{}".format(device0))


#获取数据集
train_data = torchvision.datasets.CIFAR10(root="torchvision_dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10(root="torchvision_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

#获取数据集的基本信息
#看数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度:{}".format(train_data_size))
print("测试集的长度:{}".format(test_data_size))
#看分类类别
data_classes = train_data.classes
print("数据集的分类数")
print(data_classes)
#查看图片的大小, 注意，一个train_data是包含了img和target的，这里我回看了以前的代码很久才发现
print("-------以下无序列表为从数据集下载图片下来到观察图片全过程-------")
img, target = train_data[0]
print("·确认得到一个type为：{}".format(type(img)))
print("·通过tuple分离出img和target后，获取的图片的大小为：{}".format(img.shape))
#获取一个ToPIL的transform
imgPILTransform = torchvision.transforms.ToPILImage()
imgPIL = imgPILTransform(img)
print("·这样做得到的type才是PIL.Image：{}，才可以调用show()".format(type(imgPIL)))
print("-------------------------------------------------------")

#加载数据
train_dataLoader = DataLoader(dataset=train_data, batch_size=64)
test_dataLoader = DataLoader(dataset=test_data, batch_size=64)


#加载模型
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

#创建模型
tudui = Tudui()

#调用cuda，使用GPU训练
tudui = tudui.to(device0)

#损失函数,随便选的，用交叉熵试试
loss_fn = nn.CrossEntropyLoss()

#调用cuda，使用GPU训练
loss_fn = loss_fn.to(device0)
#优化器
learningRate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learningRate)

#设置训练网络的参数
#记录训练次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter("logsForWholeTrain")

#记录运算时间
start_time = time.time()

#开始训练
for i in range(epoch):
    print("------第{}轮训练开始-----".format(i+1))

    #开始每一轮的训练
    tudui.train() #这个函数会优化某些层的计算速率，但不一定要用到，只优化部分层，详见官方文档
    for data in train_dataLoader:
        imgs, targets = data
        #为imgs、targets调用cuda方法
        imgs = imgs.to(device0)
        targets = targets.to(device0)
        ouputs = tudui(imgs)
        loss = loss_fn(ouputs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #记录训练数据
        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0 :
            #每逢100次训练记录一次
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item())) #第二个参数写loss和loss.item()都行，后者更专业化
            writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=total_train_step)

    #测试步骤开始
    tudui.eval() #这个函数会优化测试时某些层的计算速率，但不一定要用到，只优化部分层，详细见官方文档
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            # 为imgs、targets调用cuda方法
            imgs = imgs.to(device0)
            targets = targets.to(device0)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            #计算正确率,详见test_classfication.py， ，其中sum()是把True当成1，False当0，然后相加看看每一个batch里有多少个匹配正确
            #用.item()把tensor类型里面的单变量取出来，就可以得到int类型的数据了
            accuracy = (outputs.argmax(1) == targets).sum().item()
            # accuracy是一个tensor(x)，新版本已禁止tensor和int相除，所以要在继续正确率的时候，先把accuracy从tensor转为int
            total_accuracy = total_accuracy + accuracy

    print("第{}轮模型测试的测试集总Loss：{}".format(total_test_step+1, total_test_loss))
    print("第{}轮模型测试的测试集总准确率：{}".format(total_test_step + 1, total_accuracy/test_data_size))
    print("第{}轮的总运算时间为：{}".format(total_test_step + 1, time.time()-start_time))
    writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=total_test_step)
    writer.add_scalar(tag="test_accuracy", scalar_value=total_accuracy/test_data_size, global_step=total_test_step)
    total_test_step = total_test_step + 1

    #保存模型：每一个epoch都保存一个
    torch.save(tudui, "whole_train_tudui_{}.pth".format(i+1))
    #torch.save(tudui.state_dict(), "whole_train_tudui_{}".format(i+1)) 保存方式2
    print("第{}轮模型已保存".format(i+1))
writer.close()