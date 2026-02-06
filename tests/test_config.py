"""测试配置模块"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.config import Config, ModelConfig, TrainingConfig, DataConfig, PathConfig


class TestConfig:
    """测试配置类"""
    
    def test_model_config_defaults(self):
        """测试模型配置默认值"""
        config = ModelConfig()
        assert config.input_size == 784
        assert config.hidden_size == 500
        assert config.num_classes == 10
    
    def test_training_config_defaults(self):
        """测试训练配置默认值"""
        config = TrainingConfig()
        assert config.batch_size == 100
        assert config.learning_rate == 0.001
        assert config.num_epochs == 5
        assert config.log_interval == 100
    
    def test_data_config_defaults(self):
        """测试数据配置默认值"""
        config = DataConfig()
        assert config.mnist_mean == (0.1307,)
        assert config.mnist_std == (0.3081,)
    
    def test_path_config(self):
        """测试路径配置"""
        config = PathConfig()
        assert os.path.exists(config.project_root)
        assert 'data' in config.data_dir
        assert 'models' in config.models_dir
    
    def test_config_singleton(self):
        """测试统一配置类"""
        assert hasattr(Config, 'model')
        assert hasattr(Config, 'training')
        assert hasattr(Config, 'data')
        assert hasattr(Config, 'paths')
