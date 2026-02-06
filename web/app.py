import torch
import torchvision
from torchvision import transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import FNN

from flask import Flask, render_template, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FNN(input_size=784, hidden_size=500, num_classes=10).to(device)

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'fnn_mnist.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

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

def tensor_to_base64(tensor):
    tensor = tensor.squeeze()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    img = Image.fromarray(tensor.numpy().astype('uint8'), 'L')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_samples')
def get_samples():
    indices = torch.randperm(len(test_dataset))[:20]
    samples = []
    
    for idx in indices:
        image, label = test_dataset[idx]
        
        with torch.no_grad():
            image_flat = image.reshape(-1, 784).to(device)
            output = model(image_flat)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
        
        original_image = test_dataset.data[idx].numpy()
        img = Image.fromarray(original_image, 'L')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        samples.append({
            'image': img_str,
            'true_label': int(label),
            'predicted_label': int(prediction),
            'is_correct': bool(label == prediction)
        })
    
    return jsonify(samples)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
