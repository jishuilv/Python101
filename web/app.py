"""Flask Web应用 - 可视化MNIST模型推理"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

import torch
from flask import Flask, render_template, jsonify
import base64
from io import BytesIO
from PIL import Image

from config import Config
from utils import load_model, load_mnist_datasets

app = Flask(__name__)

# 全局变量 - 懒加载模式
_model = None
_device = None
_test_dataset = None


def get_model():
    """获取模型实例（单例模式）"""
    global _model, _device
    if _model is None:
        _model, _device = load_model()
    return _model, _device


def get_test_dataset():
    """获取测试数据集（单例模式）"""
    global _test_dataset
    if _test_dataset is None:
        _, _test_dataset = load_mnist_datasets()
    return _test_dataset


def image_to_base64(image_array) -> str:
    """
    将numpy数组图像转换为base64编码
    
    Args:
        image_array: numpy数组格式的图像
    
    Returns:
        base64编码的PNG图像字符串
    """
    img = Image.fromarray(image_array, 'L')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/get_samples')
def get_samples():
    """
    获取随机样本的预测结果API
    
    Returns:
        JSON格式的样本数据列表
    """
    try:
        model, device = get_model()
        test_dataset = get_test_dataset()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    
    num_samples = 20
    indices = torch.randperm(len(test_dataset))[:num_samples]
    samples = []
    
    for idx in indices:
        image, label = test_dataset[idx]
        
        with torch.no_grad():
            image_flat = image.reshape(-1, Config.model.input_size).to(device)
            output = model(image_flat)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
        
        original_image = test_dataset.data[idx].numpy()
        img_str = image_to_base64(original_image)
        
        samples.append({
            'image': img_str,
            'true_label': int(label),
            'predicted_label': int(prediction),
            'is_correct': bool(label == prediction)
        })
    
    return jsonify(samples)


if __name__ == '__main__':
    # 生产环境应该关闭debug模式
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, port=5000)
