import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json  # 新增：用于保存结果
from model import CIFAR10_CNN

def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 1. 加载测试数据
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载/加载数据集
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                                download=True, transform=test_transform)
    
    # 【关键修改】虽然这里 num_workers=2，但在 main() 函数里跑就没问题了
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 2. 加载模型
    model = CIFAR10_CNN().to(device)
    model_path = "outputs/cifar10_cnn_model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Please run train.py first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 评估循环
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    print("Evaluating on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Overall Test Accuracy: {accuracy:.2f}%")

    # === 新增：保存准确率供报告生成使用 ===
    metrics = {
        "test_accuracy": accuracy
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics saved to outputs/metrics.json")

    # 4. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png')
    print("Confusion Matrix saved to outputs/confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    main()