# 基于 CNN 的 CIFAR-10 图像分类实战

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

本项目是一个基于 PyTorch 深度学习框架的图像分类工程，使用自定义的 4 层卷积神经网络（CNN）在 CIFAR-10 数据集上进行训练与评估。项目实现了从数据预处理、模型构建、训练循环到单图推理的全流程，并包含自动化的实验报告生成脚本。

## 📂 目录结构

```text
CNN_CIFAR10_Project/
├── data/                    # 数据集存放目录 (运行脚本自动下载，不通过Git提交)
├── doc/                     # 实验文档与报告
│   └── report.md            # 自动生成的实验报告
├── outputs/                 # 输出文件 (模型权重、训练曲线、混淆矩阵、预测结果)
├── model.py                 # CNN 模型结构定义
├── train.py                 # 模型训练脚本
├── evaluate.py              # 模型评估与混淆矩阵生成
├── predict.py               # 单张图像推理脚本
├── generate_report.py       # 自动化报告生成工具
├── get_image.py             # (可选) 自动下载测试图片工具
├── requirements.txt         # 项目依赖库
└── README.md                # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

建议使用 Conda 创建虚拟环境，并安装依赖：

```bash
# 安装依赖
pip install -r requirements.txt
```

*注意：如果需要 GPU 加速，请确保安装了与您 CUDA 版本匹配的 PyTorch。*

### 2. 模型训练

运行 `train.py` 开始训练。脚本会自动下载 CIFAR-10 数据集（若未下载），并保存训练好的模型至 `outputs/` 目录。

```bash
python train.py
```
*   **默认配置**：Epochs=30, Batch_size=64, LR=0.001 (Adam)。
*   **输出**：`cifar10_cnn_model.pth`, `training_curve.png`。

### 3. 模型评估

在测试集上评估模型性能，并生成混淆矩阵。

```bash
python evaluate.py
```
*   **输出**：`metrics.json`, `confusion_matrix.png`。

### 4. 单图预测

请在项目根目录下放置一张名为 `test_image.jpg` 的图片，然后运行：

```bash
python predict.py
```
*   **输出**：控制台输出预测概率，并生成可视化的 `single_prediction.png`。

### 5. 生成实验报告

上述步骤完成后，运行以下脚本即可自动生成包含图表和数据的 Markdown 实验报告：

```bash
python generate_report.py
```
报告将生成于 `doc/report.md`。

## 🧠 模型架构

本项目采用轻量级 CNN 结构：
1.  **Conv Block 1**: 3→16 channels, 3x3 Conv, ReLU, MaxPool
2.  **Conv Block 2**: 16→32 channels, 3x3 Conv, ReLU, MaxPool
3.  **Conv Block 3**: 32→64 channels, 3x3 Conv, ReLU, MaxPool
4.  **Conv Block 4**: 64→128 channels, 3x3 Conv, ReLU, MaxPool
5.  **Classifier**: Flatten -> Linear(512) -> ReLU -> Dropout(0.5) -> Linear(10)

## 📊 实验结果

*   **测试集准确率**: ~78% - 80% (30 Epochs)
*   **优化策略**: 采用了 RandomCrop/HorizontalFlip 数据增强与 StepLR 学习率衰减策略，有效防止了过拟合。

## 📝 许可证

MIT License

