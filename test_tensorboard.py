from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


writer = SummaryWriter("logs")

#绘制坐标图
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)
for i in range(100):
    writer.add_scalar("y=x",i,i)

#绘制图片
image_path = "dataset/train/ants/0013035.jpg"
img = Image.open(image_path)
img_array = np.array(img)
print(img_array.shape)
writer.add_image("Image1",img_array,1,dataformats='HWC')

image_path2 = "dataset/train/ants/5650366_e22b7e1065.jpg"
img2 = Image.open(image_path2)
img_array2 = np.array(img2)
print(img_array2.shape)
writer.add_image("Image1",img_array2,2,dataformats='HWC')

writer.close()