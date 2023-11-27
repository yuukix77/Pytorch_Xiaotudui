import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("torchvision_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

#看一下第一张图片和它的target
print(test_data.classes)
img, target = test_data[0]
print(img.shape)
print(target)


writer = SummaryWriter("logsForTestDataLoader")
writerAddImageStep = 0

# #取出每一个test_loader的返回
# for data in test_loader:
#     imgs, targets = data
#     # print(imgs.shape)
#     # print(targets)
#     writer.add_images("test_data_dropLastTrue", imgs, writerAddImageStep)
#     writerAddImageStep = writerAddImageStep + 1

#Epoch:下一把牌，重新开始在数据集中摸牌，False则打乱顺序，True则不打乱,通过改变shuffle参数调整
for epoch in range(2):
    writerAddImageStep = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("test_data_shufferTrue,epoch:{}".format(epoch), imgs, writerAddImageStep)
        writerAddImageStep = writerAddImageStep + 1

writer.close()