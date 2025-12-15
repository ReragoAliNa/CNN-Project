import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import CIFAR10_CNN
import os

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict_single_image(image_path, model_path):
    # 1. 准备模型
    model = CIFAR10_CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # 确保尺寸一致
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image {image_path} not found.")
        return

    input_tensor = transform(image).unsqueeze(0).to(device) # 增加 Batch 维度

    # 3. 预测
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prob, pred_class_idx = torch.max(probabilities, 1)
        
        pred_class = class_names[pred_class_idx.item()]
        pred_prob = prob.item()

    # 4. 打印结果
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {pred_prob:.2%}")

    # 5. 显示
    plt.figure(figsize=(5, 5))
    plt.imshow(image) # 显示原图
    plt.title(f"Pred: {pred_class} ({pred_prob:.2%})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    img_path = "test_image.jpg" # 请确保根目录下有这张图
    model_file = "outputs/cifar10_cnn_model.pth"
    
    if os.path.exists(model_file):
        predict_single_image(img_path, model_file)
    else:
        print("Model file not found. Run train.py first.")