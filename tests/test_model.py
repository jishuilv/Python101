"""测试模型模块"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from src.model import FNN
from src.config import Config


class TestFNN:
    """测试FNN模型类"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = FNN(
            input_size=Config.model.input_size,
            hidden_size=Config.model.hidden_size,
            num_classes=Config.model.num_classes
        )
        
        assert model is not None
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'relu')
        assert hasattr(model, 'fc2')
    
    def test_forward_pass(self):
        """测试前向传播"""
        model = FNN(
            input_size=Config.model.input_size,
            hidden_size=Config.model.hidden_size,
            num_classes=Config.model.num_classes
        )
        
        batch_size = 32
        dummy_input = torch.randn(batch_size, Config.model.input_size)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (batch_size, Config.model.num_classes)
    
    def test_model_to_device(self):
        """测试模型移到设备"""
        device = torch.device('cpu')
        model = FNN(
            input_size=Config.model.input_size,
            hidden_size=Config.model.hidden_size,
            num_classes=Config.model.num_classes
        ).to(device)
        
        assert next(model.parameters()).device == device
