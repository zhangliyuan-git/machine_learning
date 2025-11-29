import numpy as np
import torch
from sklearn.decomposition import PCA

# 表示一个人的特征向量（年龄，收入，身高）
vector_np = np.array([28, 15000, 178])
vector_pt = torch.tensor([28, 15000, 178])

#表示三个人的数据集（矩阵）
matrix_np = np.array([[28, 15000, 178], [25, 12000, 165], [30, 18000, 182]])
print(matrix_np)
# 表示一个批次的2张3x3的RGB图片（张量）
batch_of_images = torch.randn(2, 3, 3, 4) # [批次， 高度， 宽度， 通道]
print(batch_of_images)
# 模拟一个神经网络层：输出=输入 x 权重 + 偏置
input_data = torch.randn(100, 784)  # 100张展平后的784维图片
weight = torch.randn(784, 128)  # 权重矩阵
bias = torch.randn(128)  # 偏置向量

# 一次前向传播（矩阵乘法+广播加法）
output = torch.matmul(input_data, weight) + bias
print(output.shape)

# 应用ReLu激活函数
activated_output = torch.relu(output)
print(activated_output)
# 生成模拟数据
data = np.random.randn(100, 5) # 100个样本，5个特征

# 用PCA降维到2个主成分，其背后是特征值分解
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

print(f"原始形状: {data.shape}")   # (100, 5)
print(f"降维后形状: {reduced_data.shape}") # (100, 2)
print(f"各主成分的方差贡献: {pca.explained_variance_ratio_}")