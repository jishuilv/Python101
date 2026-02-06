"""工具类模块 - 提供数据加载、模型管理等通用功能"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import os

from config import Config
from model import FNN


def get_device() -> torch.device:
    """获取计算设备（优先使用GPU）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_transform() -> transforms.Compose:
    """获取MNIST数据预处理变换"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(Config.data.mnist_mean, Config.data.mnist_std)
    ])


def load_mnist_datasets(download: bool = True) -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    """
    加载MNIST训练集和测试集
    
    Args:
        download: 如果数据集不存在是否下载
    
    Returns:
        (train_dataset, test_dataset)
    """
    transform = get_data_transform()
    data_dir = Config.paths.data_dir
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=download
    )
    
    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: torchvision.datasets.MNIST,
    test_dataset: torchvision.datasets.MNIST,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Args:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        batch_size: 批量大小，默认使用配置中的值
    
    Returns:
        (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.training.batch_size
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def load_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[FNN, torch.device]:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径，默认使用配置中的路径
        device: 计算设备，默认自动选择
    
    Returns:
        (model, device)
    
    Raises:
        FileNotFoundError: 模型文件不存在
    """
    if device is None:
        device = get_device()
    
    if model_path is None:
        model_path = Config.paths.default_model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = FNN(
        input_size=Config.model.input_size,
        hidden_size=Config.model.hidden_size,
        num_classes=Config.model.num_classes
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    return model, device


def save_model(model: FNN, model_path: Optional[str] = None) -> str:
    """
    保存模型权重
    
    Args:
        model: 要保存的模型
        model_path: 保存路径，默认使用配置中的路径
    
    Returns:
        保存的文件路径
    """
    if model_path is None:
        model_path = Config.paths.default_model_path
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    return model_path
