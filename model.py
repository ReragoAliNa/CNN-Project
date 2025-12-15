import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        # 卷积块 1: 3 -> 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32->16

        # 卷积块 2: 16 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16->8

        # 卷积块 3: 32 -> 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8->4

        # 卷积块 4: 64 -> 128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 4->2

        # 全连接层
        # 展平后维度: 128通道 * 2高 * 2宽
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.dropout = nn.Dropout(0.5) # 防止过拟合
        self.fc2 = nn.Linear(512, 10)  # 输出10个类别

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # 展平
        x = x.view(-1, 128 * 2 * 2)
        
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x