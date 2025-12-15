import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from model import CIFAR10_CNN

def main():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出文件夹
    os.makedirs("outputs", exist_ok=True)

    # 2. 数据预处理与增强
    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 验证/测试集仅归一化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 3. 加载数据集
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                    download=True, transform=train_transform)

    # 划分训练集(45000) 和 验证集(5000)
    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    batch_size = 64
    
    # 【关键修改】在 Windows 下，num_workers > 0 必须配合 if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. 初始化模型、损失函数、优化器
    model = CIFAR10_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 5. 训练循环
    num_epochs = 30
    train_losses, val_losses, val_accs = [], [], []

    print("Start Training...")
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.2f}%")

    # 6. 保存模型
    torch.save(model.state_dict(), "outputs/cifar10_cnn_model.pth")
    print("Model saved to outputs/cifar10_cnn_model.pth")

    # 7. 可视化
    plt.figure(figsize=(12, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accs, label='Val Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.savefig('outputs/training_curve.png')
    print("Training curve saved to outputs/training_curve.png")
    plt.show()

# 这一行是解决 Windows 多进程报错的关键
if __name__ == '__main__':
    main()