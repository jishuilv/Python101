"""项目配置管理模块"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """神经网络模型配置"""
    input_size: int = 784
    hidden_size: int = 500
    num_classes: int = 10


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 100
    learning_rate: float = 0.001
    num_epochs: int = 5
    log_interval: int = 100


@dataclass
class DataConfig:
    """数据配置"""
    mnist_mean: Tuple[float] = (0.1307,)
    mnist_std: Tuple[float] = (0.3081,)


@dataclass
class PathConfig:
    """路径配置"""
    project_root: str = None
    
    def __post_init__(self):
        if self.project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_root, 'data')
    
    @property
    def models_dir(self) -> str:
        return os.path.join(self.project_root, 'models')
    
    @property
    def default_model_path(self) -> str:
        return os.path.join(self.models_dir, 'fnn_mnist.pth')


class Config:
    """统一配置类"""
    model = ModelConfig()
    training = TrainingConfig()
    data = DataConfig()
    paths = PathConfig()
