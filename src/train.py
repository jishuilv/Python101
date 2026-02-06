"""MNIST模型训练脚本"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from model import FNN
from utils import (
    get_device,
    load_mnist_datasets,
    create_dataloaders,
    save_model
)


def train_one_epoch(
    model: FNN,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> None:
    """
    训练一个epoch
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前epoch编号
        total_epochs: 总epoch数
    """
    model.train()
    total_step = len(train_loader)
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, Config.model.input_size).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % Config.training.log_interval == 0:
            print(f'Epoch [{epoch + 1}/{total_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')


def evaluate_model(
    model: FNN,
    test_loader,
    device: torch.device
) -> float:
    """
    在测试集上评估模型
    
    Args:
        model: 神经网络模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        测试准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, Config.model.input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main() -> None:
    """主训练函数"""
    device = get_device()
    print(f'Using device: {device}')
    
    print('Loading datasets...')
    train_dataset, test_dataset = load_mnist_datasets()
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset)
    
    model = FNN(
        input_size=Config.model.input_size,
        hidden_size=Config.model.hidden_size,
        num_classes=Config.model.num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.training.learning_rate)
    
    print('Starting training...')
    for epoch in range(Config.training.num_epochs):
        train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, Config.training.num_epochs
        )
    
    print('Evaluating model...')
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy of the model on the 10000 test images: {accuracy:.2f} %')
    
    print('Saving model...')
    model_path = save_model(model)
    print(f'Model saved as {model_path}')


if __name__ == '__main__':
    main()
