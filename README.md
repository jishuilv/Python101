# MNIST手写数字识别

基于PyTorch的前馈神经网络（FNN）实现MNIST手写数字识别任务。

## 项目概述

本项目使用PyTorch构建一个简单的前馈神经网络，用于识别MNIST数据集中的手写数字（0-9）。

## 环境要求

- Python 3.x
- PyTorch
- torchvision

## 安装依赖

```bash
pip install torch torchvision
```

## 文件结构

```
Python101/
├── FNN.py              # 训练脚本
├── inference.py        # 推理/验证脚本
├── README.md           # 项目说明文档
├── fnn_mnist.pth       # 训练好的模型权重
└── dataset/
    └── MNIST/          # MNIST数据集（自动下载）
```

## 快速开始

### 1. 训练模型

```bash
python FNN.py
```

这会：
- 加载MNIST数据集
- 训练5个epoch
- 显示训练过程和损失值
- 计算测试准确率
- 保存模型为 `fnn_mnist.pth`

### 2. 验证模型能力

```bash
python inference.py
```

这会：
- 加载已训练好的模型
- 计算整体准确率
- 随机选择15个样本展示预测结果

## 模型架构

- **输入层**: 784个神经元（28×28像素展平）
- **隐藏层**: 500个神经元 + ReLU激活函数
- **输出层**: 10个神经元（对应0-9十个数字）

## 训练配置

- **批量大小**: 100
- **学习率**: 0.001
- **优化器**: Adam
- **损失函数**: 交叉熵损失
- **训练轮数**: 5 epochs

## 性能结果

- **测试准确率**: 97.52%（在10,000张测试图像上）

## 使用模型

训练完成后，模型权重会保存为 `fnn_mnist.pth`，可以使用以下方式加载：

```python
import torch
from FNN import FNN

model = FNN(input_size=784, hidden_size=500, num_classes=10)
model.load_state_dict(torch.load('fnn_mnist.pth'))
model.eval()
```
