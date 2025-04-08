import numpy as np
import torch
import matplotlib.pyplot as plt

import os#设置环境变量，没有这个就会报错
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 加载数据
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 输入特征
y_data = torch.from_numpy(xy[:, [-1]])  # 目标值


# 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 第一层：输入特征维度为 8，输出特征维度为 6
        self.linear2 = torch.nn.Linear(6, 4)  # 第二层：输入特征维度为 6，输出特征维度为 4
        self.linear3 = torch.nn.Linear(4, 1)  # 第三层：输入特征维度为 4，输出特征维度为 1
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))  # 第一层：线性变换后接 Sigmoid 激活函数
        x = self.sigmoid(self.linear2(x))  # 第二层：线性变换后接 Sigmoid 激活函数
        x = self.sigmoid(self.linear3(x))  # 第三层：线性变换后接 Sigmoid 激活函数
        return x


# 初始化模型、损失函数和优化器
model = Model()
criterion = torch.nn.BCELoss(size_average=True)  # 二元交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # SGD 优化器

# 用于存储每个 epoch 的损失值
losses = []

# 训练过程
for epoch in range(100):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 存储损失值
    losses.append(loss.item())

    # 反向传播和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 可视化损失值
plt.plot(range(100), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid()
plt.show()