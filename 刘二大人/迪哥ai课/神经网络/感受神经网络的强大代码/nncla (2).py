import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理：归一化到 [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# 2. 构建神经网络模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 将 28x28 的图像展平为 784 维向量
    layers.Dense(128, activation='relu'),  # 全连接层，128 个神经元，ReLU 激活函数
    layers.Dropout(0.2),                  # Dropout 层，防止过拟合
    layers.Dense(10, activation='softmax')  # 输出层，10 个类别，Softmax 激活函数
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 训练模型
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# 4. 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# 5. 可视化训练过程
plt.figure(figsize=(12, 4))

# 绘制训练和验证的准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制训练和验证的损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 6. 预测并可视化结果
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# 随机选择 10 个测试样本进行可视化
indices = np.random.choice(len(test_images), 10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, idx in enumerate(indices):
    ax = axes[i // 5, i % 5]
    ax.imshow(test_images[idx], cmap='gray')
    ax.set_title(f"True: {test_labels[idx]}\nPred: {predicted_labels[idx]}")
    ax.axis('off')
plt.tight_layout()
plt.show()