import torch
import torchvision
from torchvision import transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import FNN

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'fnn_mnist.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNN(input_size=784, hidden_size=500, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def show_sample_predictions(model, device, num_samples=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )
    
    indices = torch.randperm(len(test_dataset))[:num_samples]
    
    print(f'\n=== 展示 {num_samples} 个样本的预测结果 ===')
    print('-' * 50)
    
    correct_count = 0
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        
        with torch.no_grad():
            image_flat = image.reshape(-1, 784).to(device)
            output = model(image_flat)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
        
        is_correct = label == prediction
        if is_correct:
            correct_count += 1
        
        status = '✓ 正确' if is_correct else '✗ 错误'
        print(f'样本 {i + 1}: 真实值 = {label}, 预测值 = {prediction}  [{status}]')
    
    print('-' * 50)
    print(f'本次抽样准确率: {100 * correct_count / num_samples:.2f}%')

def calculate_accuracy(model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 784).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\n模型在10000张测试图像上的准确率: {accuracy:.2f}%')
    return accuracy

def main():
    print('正在加载训练好的模型...')
    model, device = load_model()
    print(f'使用设备: {device}')
    
    print('\n=== 计算整体准确率 ===')
    calculate_accuracy(model, device)
    
    print('\n=== 展示样本预测 ===')
    show_sample_predictions(model, device, num_samples=15)

if __name__ == '__main__':
    main()
