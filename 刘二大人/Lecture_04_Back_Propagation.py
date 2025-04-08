import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.Tensor([1.0,1.0])
w.requires_grad = True
b=torch.Tensor([1.0])
b.requires_grad=True

# 定义前向传播函数
def forward(x):
    # 将 x 转换为张量
    x = torch.tensor([x], dtype=torch.float32)
    # 计算 x^2 和 x
    x_squared = x**2
    x_linear = x
    # 计算 y = x^2 * w1 + x * w2 + b
    y_pred = x_squared * w[0] + x_linear * w[1] + b
    return y_pred

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)",  4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad,b.grad)
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()
        print("progress:", epoch, l.item())
print("predict (after training)", 4, forward(4).item())