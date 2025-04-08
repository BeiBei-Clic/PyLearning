import torch

'''x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)'''

import torch
import torch.nn as nn
import torch.optim as optim

# 数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 定义模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# 损失函数
criterion = nn.MSELoss(reduction='sum')  # 使用 reduction='sum'

# 测试数据
x_test = torch.Tensor([[4.0]])

# 定义优化器列表
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "Adagrad":optim.Adagrad,
    "Adamax":optim.Adamax,
    "ASGD":optim.ASGD
}

# 存储结果
results = {}

# 训练和评估每个优化器
for name, optimizer_class in optimizers.items():
    # 重新初始化模型
    model = LinearModel()

    # 初始化优化器
    optimizer = optimizer_class(model.parameters(), lr=0.01)

    # 训练过程
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 保存结果
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    y_test_pred = model(x_test).item()

    results[name] = {
        "w": w,
        "b": b,
        "y_pred": y_test_pred,
        "loss": loss.item()
    }

# 打印结果
for name, result in results.items():
    print(f"Optimizer: {name}")
    print(f"  w = {result['w']:.4f}")
    print(f"  b = {result['b']:.4f}")
    print(f"  y_pred (x=4.0) = {result['y_pred']:.4f}")
    print(f"  Final Loss: {result['loss']:.4f}")
    print("-" * 30)