from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import  transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/20935278_9190345f6b.jpg")

##ToTensor:把图片转换为三维的H-W-RGB模式
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Example1",img_tensor,1)

##Normalize：归一化
# 人为设置一个均值和标准差给他先，处理公式是如下
# output[channel] = (input[channel] - mean[channel]) / std[channel]
# 参数都填0.5的原因是：假设input范围在[0-1]，用Noramalize参数填0.5，就能把这个范围转换到[-1,1]
print("归一化前第一个像素的样子：" , img_tensor[0][0][0]) #打印第一个数据看看，为下面归一化做准备
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print("归一化后第一个像素的样子：" , img_norm[0][0][0]) #打印第一个数据看看，看看归一化做了什么
writer.add_image("NormalizeExample1",img_norm,2)

##Resize: 一个参数就等比缩放,两个就H，W缩放
print("图片原大小：", img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img) #参数是PIL.Image
print("resize1:" , img_resize)
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("ResizeExample1",img_resize_tensor,1)

##Compose: 流水线化我们的工程,例如把Resize 和 ToTensor 组合在一起
trans_resize_2 = transforms.Resize((100,50))
print("resize2:", trans_resize_2(img))
# compose的输入是PIL.Image，然后经过resize之后把resize的output传到totensor
# 简化理解：PIL.Image -> PIL.Image(Reshape) -> ToTensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_composeProcess_tensor = trans_compose(img)
writer.add_image("ComposeExample1",img_composeProcess_tensor,1)

##RandomCrop: 随机裁剪

trans_random = transforms.RandomCrop(100) #按固定长宽裁剪，我第一次写了512，但其实图片只有500*375，所以一直报错
#trans_random2 = transforms.RandomCrop(100,200) #按HW裁剪
trans_compose2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose2(img)
    writer.add_image("RandomCropExample1", img_crop, i)

writer.close()

