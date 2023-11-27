import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="torchvision_dataset", train=True,
                                         transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="torchvision_dataset", train=False,
                                         transform=dataset_transform, download=True)
##看一下这几个变量长什么样，并最终show出图片
# print(test_set[0])
# #classes 是类别的意思，输出3代表是小猫
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

#确认是tensor类型
print(test_set[0])

writer = SummaryWriter("logsForTestDataset")
for i in range(len(test_set)):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()








