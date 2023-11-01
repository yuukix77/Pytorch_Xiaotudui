from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import  transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/20935278_9190345f6b.jpg")

##ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Example1",img_tensor)
writer.close()

