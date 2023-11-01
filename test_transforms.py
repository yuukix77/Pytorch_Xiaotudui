from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import  Image

###先获取图片地址->打开图片->转为array/tensor->添加到writer

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

## 使用ToTensor读取图片并转换为tensor

#这是一个totensor对象
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

#写到tensorboard
writer.add_image("标题",tensor_img)
writer.close()


