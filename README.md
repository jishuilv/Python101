# MNIST手写数字识别

基于PyTorch的前馈神经网络（FNN）实现MNIST手写数字识别任务。

## 项目概述

本项目使用PyTorch构建一个简单的前馈神经网络，用于识别MNIST数据集中的手写数字（0-9）。提供了三种验证方式：命令行推理、可视化Web界面。

## 环境要求

- Python 3.x
- PyTorch
- torchvision
- Flask
- Pillow

## 安装依赖

```bash
pip install torch torchvision flask pillow
```

## 项目结构

```
Python101/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── model.py            # 模型定义
│   ├── train.py            # 训练脚本
│   └── inference.py        # 命令行推理脚本
├── web/                    # Web应用目录
│   ├── app.py             # Flask后端
│   └── templates/
│       └── index.html      # Web界面前端
├── models/                 # 保存的模型
│   └── fnn_mnist.pth
├── data/                   # 数据集
│   └── MNIST/
└── README.md
```

## 快速开始

### 1. 训练模型

```bash
cd src
python train.py
```

这会：
- 加载MNIST数据集
- 训练5个epoch
- 显示训练过程和损失值
- 计算测试准确率
- 保存模型到 `models/fnn_mnist.pth`

### 2. 命令行验证模型

```bash
cd src
python inference.py
```

这会：
- 加载已训练好的模型
- 计算整体准确率
- 随机选择15个样本展示预测结果

### 3. 可视化Web界面（推荐）⭐

启动Web应用：

```bash
cd web
python app.py
```

然后在浏览器中打开：http://localhost:5000

**Web界面功能：**
- 🖼️ 直观展示手写数字原始图像
- ✅ 同时显示真实值和预测值
- 🎨 用颜色区分正确/错误识别
- 📊 显示本次抽样准确率统计
- 🔄 点击按钮加载新的随机样本

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

训练完成后，模型权重会保存为 `models/fnn_mnist.pth`，可以使用以下方式加载：

```python
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.model import FNN

model = FNN(input_size=784, hidden_size=500, num_classes=10)
model.load_state_dict(torch.load('models/fnn_mnist.pth'))
model.eval()
```
