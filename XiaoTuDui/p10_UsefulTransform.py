from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#ToTensor使用
writer=SummaryWriter("logs")
img=Image.open("image/咖喱鱼蛋.jpg")
# print(img)

trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor,0)
# writer.close()

#归一化Normalize
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#均值和标准差
img_norm=trans_norm(img_tensor)
writer.add_image("Normalize",img_norm,0)

#Resize
# print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
# print(img_resize)

#Compose-Resize-2
trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])#将多个图像变换的操作结合在一起
# print(type(img))
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#随机裁剪RandomCrop
trans_random=transforms.RandomCrop((512,860))
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCropHW",img_crop,i)


writer.close()


