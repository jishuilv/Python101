"""命令行推理脚本 - 验证模型能力"""

import torch

from config import Config
from utils import (
    get_device,
    load_mnist_datasets,
    create_dataloaders,
    load_model
)


def calculate_accuracy(model, device) -> float:
    """
    计算模型在测试集上的准确率
    
    Args:
        model: 神经网络模型
        device: 计算设备
    
    Returns:
        准确率（百分比）
    """
    _, test_dataset = load_mnist_datasets()
    _, test_loader = create_dataloaders(test_dataset, test_dataset)
    
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


def show_sample_predictions(model, device, num_samples: int = 10) -> float:
    """
    展示随机样本的预测结果
    
    Args:
        model: 神经网络模型
        device: 计算设备
        num_samples: 展示的样本数量
    
    Returns:
        抽样准确率（百分比）
    """
    _, test_dataset = load_mnist_datasets()
    
    indices = torch.randperm(len(test_dataset))[:num_samples]
    
    print(f'\n=== 展示 {num_samples} 个样本的预测结果 ===')
    print('-' * 50)
    
    correct_count = 0
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        
        with torch.no_grad():
            image_flat = image.reshape(-1, Config.model.input_size).to(device)
            output = model(image_flat)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
        
        is_correct = label == prediction
        if is_correct:
            correct_count += 1
        
        status = '✓ 正确' if is_correct else '✗ 错误'
        print(f'样本 {i + 1}: 真实值 = {label}, 预测值 = {prediction}  [{status}]')
    
    print('-' * 50)
    sample_accuracy = 100 * correct_count / num_samples
    print(f'本次抽样准确率: {sample_accuracy:.2f}%')
    
    return sample_accuracy


def main() -> None:
    """主推理函数"""
    print('正在加载训练好的模型...')
    try:
        model, device = load_model()
    except FileNotFoundError as e:
        print(f'错误: {e}')
        print('请先运行 train.py 训练模型')
        return
    
    print(f'使用设备: {device}')
    
    print('\n=== 计算整体准确率 ===')
    accuracy = calculate_accuracy(model, device)
    print(f'模型在10000张测试图像上的准确率: {accuracy:.2f}%')
    
    print('\n=== 展示样本预测 ===')
    show_sample_predictions(model, device, num_samples=15)


if __name__ == '__main__':
    main()
