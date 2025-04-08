from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path="Dataset/data/train/ants_image/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

tensor_trans=transforms.ToTensor()#先创建一个类
tensor_img=tensor_trans(img)#调用类的__call__方法

writer.add_image("Tensor_img",tensor_img)

writer.close()

