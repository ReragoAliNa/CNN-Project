import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

# 定义您希望保存数据集的本地目录
# 程序将在当前工作目录下创建一个名为 'cifar10_data' 的文件夹
data_dir = './cifar10_data' 

# 确保目标目录存在
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")

# 定义一个简单的数据预处理步骤（将图像转换为 PyTorch Tensor）
transform = transforms.ToTensor()

print(f"Starting download of CIFAR-10 dataset to: {data_dir}")

# 使用 datasets.CIFAR10 加载训练集
# download=True 参数会自动触发下载操作
train_dataset = datasets.CIFAR10(root=data_dir, 
                                 train=True, 
                                 download=True, 
                                 transform=transform)

# 使用 datasets.CIFAR10 加载测试集
test_dataset = datasets.CIFAR10(root=data_dir, 
                                train=False, 
                                download=True, 
                                transform=transform)

print("-" * 30)
print("✅ Download and loading complete!")
print(f"训练集大小: {len(train_dataset)} 张图片")
print(f"测试集大小: {len(test_dataset)} 张图片")
print(f"数据已保存至本地目录: {os.path.abspath(data_dir)}")