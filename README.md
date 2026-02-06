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
- pytest (可选，用于运行测试)

## 安装依赖

```bash
pip install torch torchvision flask pillow pytest
```

## 项目结构

```
Python101/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── config.py          # 配置管理模块
│   ├── model.py           # 模型定义
│   ├── utils.py           # 工具类模块
│   ├── train.py           # 训练脚本
│   └── inference.py       # 命令行推理脚本
├── web/                    # Web应用目录
│   ├── app.py             # Flask后端
│   └── templates/
│       └── index.html      # Web界面前端
├── tests/                  # 单元测试
│   ├── __init__.py
│   ├── test_config.py     # 配置模块测试
│   └── test_model.py      # 模型测试
├── models/                 # 保存的模型
│   └── fnn_mnist.pth
├── data/                   # 数据集
│   └── MNIST/
├── README.md               # 项目说明
└── CODE_REVIEW.md          # 代码审查报告
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

### 4. 运行单元测试

```bash
pytest tests/ -v
```

## 代码质量

本项目已通过全面的代码审查，改进包括：

✅ **代码规范与可读性**
- 遵循单一职责原则
- 完整的文档字符串
- 消除魔法数字

✅ **设计模式与架构**
- 遵循DRY原则
- 配置与逻辑分离
- 统一的工具接口

✅ **健壮性与测试**
- 完善的错误处理
- 安全的模型加载
- 核心模块单元测试

✅ **性能与安全**
- 懒加载模式
- 可配置的debug模式
- 安全的路径处理

详细的审查报告请查看：`CODE_REVIEW.md`

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
from src.utils import load_model

# 方式1：使用utils模块
model, device = load_model()

# 方式2：手动加载
model = FNN(input_size=784, hidden_size=500, num_classes=10)
model.load_state_dict(torch.load('models/fnn_mnist.pth', weights_only=True))
model.eval()
```
