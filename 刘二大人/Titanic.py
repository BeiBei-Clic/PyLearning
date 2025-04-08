import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 加载数据
train_df = pd.read_csv('Data/titanic/train.csv')
test_df = pd.read_csv('Data/titanic/test.csv')

# 选择特征和目标变量
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

# 处理缺失值和编码分类变量
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 使用中位数填充缺失值
    ('scaler', StandardScaler())                    # 对数值特征进行标准化处理
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 使用固定值填充缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                      # 进行独热编码
])

# 定义 ColumnTransformer，整合数值特征和分类特征的预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # 数值特征的预处理管道
        ('cat', categorical_transformer, categorical_features)  # 分类特征的预处理管道
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(X)

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn


class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(input_dim, 128)
        # 第一个隐藏层到第二个隐藏层
        self.fc2 = nn.Linear(128, 256)
        # 第二个隐藏层到第三个隐藏层
        self.fc3 = nn.Linear(256, 128)
        # 第三个隐藏层到第四个隐藏层
        self.fc4 = nn.Linear(128, 64)
        # 第四个隐藏层到输出层
        self.fc5 = nn.Linear(64, 1)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一层
        x = self.relu(self.fc1(x))
        # 第二层
        x = self.sigmoid(self.fc2(x))
        # 第三层
        x = self.sigmoid(self.fc3(x))
        # 第四层
        x = self.sigmoid(self.fc4(x))
        # 输出层
        x = self.sigmoid(self.fc5(x))
        return x

# 获取输入维度
input_dim = X_processed.shape[1]
model = TitanicModel(input_dim)

# 转换为 PyTorch 张量
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
    val_preds = (val_outputs > 0.5).float()
    accuracy = (val_preds == y_val).float().mean().item()
    print(f'Validation Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.4f}')

# 预处理测试集
X_test = preprocessor.transform(test_df[features])
X_test = torch.tensor(X_test, dtype=torch.float32)

# 预测测试集
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_preds = (test_outputs > 0.5).int().numpy()

# 生成提交文件
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_preds.flatten()})
submission.to_csv('submission.csv', index=False)

import matplotlib.pyplot as plt

losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()