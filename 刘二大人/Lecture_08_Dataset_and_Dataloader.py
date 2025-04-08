import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 加载数据文件
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # 获取数据集的长度（样本数量）
        self.len = xy.shape[0]
        # 将输入特征数据转换为 PyTorch 张量
        self.x_data = torch.from_numpy(xy[:, :-1])
        # 将目标值数据转换为 PyTorch 张量
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        # 根据索引返回单个样本及其标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回数据集的长度（样本数量）
        return self.len

dataset = DiabetesDataset('Data/diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义第一个全连接层，输入特征维度为 8，输出特征维度为 6
        self.linear1 = torch.nn.Linear(8, 6)
        # 定义第二个全连接层，输入特征维度为 6，输出特征维度为 4
        self.linear2 = torch.nn.Linear(6, 4)
        # 定义第三个全连接层，输入特征维度为 4，输出特征维度为 1
        self.linear3 = torch.nn.Linear(4, 1)
        # 定义 Sigmoid 激活函数
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 第一层：线性变换后接 Sigmoid 激活函数
        x = self.sigmoid(self.linear1(x))
        # 第二层：线性变换后接 Sigmoid 激活函数
        x = self.sigmoid(self.linear2(x))
        # 第三层：线性变换后接 Sigmoid 激活函数
        x = self.sigmoid(self.linear3(x))
        # 返回最终的输出
        return x

if __name__=="__main__":
    model = Model()
    criterion = torch.nn.BCELoss(size_average=True)  # 二元交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

    for epoch in range(100):  # 训练 100 个 epoch
        for i, data in enumerate(train_loader, 0):  # 遍历 DataLoader
            # 1. Prepare data
            inputs, labels = data  # 获取输入数据和目标值
            # 2. Forward
            y_pred = model(inputs)  # 前向传播，计算预测值
            loss = criterion(y_pred, labels)  # 计算损失值
            print(epoch, i, loss.item())  # 打印当前 epoch、batch 和损失值
            # 3. Backward
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播，计算梯度
            # 4. Update
            optimizer.step()  # 更新模型参数