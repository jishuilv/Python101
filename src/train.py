import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from model import FNN
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    input_size = 784
    hidden_size = 500
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )

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

    model = FNN(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f} %')

    model_path = os.path.join(model_dir, 'fnn_mnist.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved as {model_path}')

if __name__ == '__main__':
    main()
