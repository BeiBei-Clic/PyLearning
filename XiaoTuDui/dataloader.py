import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备的测试数据集
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())#Train为False则加载测试集，Train为True则加载训练集

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

img,target=test_data[0]
# print(img.shape)
# print(target)

writer=SummaryWriter("dataloader")#指定日志文件的存储路径。在这个例子中，日志文件会被保存到当前目录下的 dataloader 文件夹中。如果文件夹不存在，TensorBoard 会自动创建它。
step=0

for data in test_loader:
    imgs,targets=data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_data_droplast",imgs,step)
    step+=1

writer.close()