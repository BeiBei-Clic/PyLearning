import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


import os#设置环境变量，没有这个就会报错
os.environ['KMP_DUPLICATE_LIB_OK']='True'



x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
 #-------------------------------------------------------#
class LogisticRegressionModel(torch.nn.Module):
     def __init__(self):
         super(LogisticRegressionModel, self).__init__()
         self.linear = torch.nn.Linear(1, 1)
     def forward(self, x):
         y_pred = F.sigmoid(self.linear(x))
         return y_pred

model = LogisticRegressionModel()
#-------------------------------------------------------#
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#-------------------------------------------------------#
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# 使用 NumPy 生成一个从 0 到 10 的线性间隔数组，包含 200 个点
x = np.linspace(0, 10, 200)
# 将 NumPy 数组转换为 PyTorch 张量，并将其形状调整为 (200, 1)，以匹配模型的输入要求
x_t = torch.Tensor(x).view((200, 1))
# 使用模型对输入数据 x_t 进行预测，得到预测结果 y_t
y_t = model(x_t)
# 将预测结果 y_t 转换为 NumPy 数组，以便用于绘图
y = y_t.data.numpy()
# 使用 Matplotlib 绘制模型预测结果的曲线
plt.plot(x, y)
# 在图上绘制一条红色的水平参考线，表示 y=0.5 的位置
# 这条线从 x=0 到 x=10，y 值始终为 0.5
plt.plot([0, 10], [0.5, 0.5], c='r')
# 设置图像的 x 轴标签为 "Hours"
plt.xlabel('Hours')
# 设置图像的 y 轴标签为 "Probability of Pass"
plt.ylabel('Probability of Pass')
# 添加网格线，以便更清晰地观察图像
plt.grid()
# 显示图像
plt.show()