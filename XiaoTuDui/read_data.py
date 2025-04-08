from conda.exports import root_dir
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, idx):#定义了如何通过索引访问对象
        img_name=self.img_path[idx]#图片文件名
        img_item_path=os.path.join(self.root_dir,self.label_dir)#相对路径
        img=Image.open(img_item_path)#读取文件
        label=self.label_dir
        return img,label

    def __len__(self):
        return  len(self.img_path)

root_dir="数据集/hymenoptera_data/train"
ants_label_dir="ants"
ants_dataset=MyData(root_dir,ants_label_dir)
bees_label_dir="bees"
bees_dataset=MyData(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset