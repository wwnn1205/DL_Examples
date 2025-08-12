import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet-5输入为32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 输入1通道，输出6通道
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入6通道，输出16通道

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16*5*5是卷积后的特征维度
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10个输出，对应0-9数字

        # 激活函数和池化层
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # LeNet-5原始使用tanh或sigmoid
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化

    def forward(self, x):
        # 第一层卷积: 卷积 -> 激活 -> 池化
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        # 第二层卷积: 卷积 -> 激活 -> 池化
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        # 展平特征图
        x = x.view(-1, 16 * 5 * 5)

        # 全连接层
        x = self.fc1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x = self.tanh(x)

        x = self.fc3(x)  # 输出层不需要激活函数，因为会用交叉熵损失

        return x

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每100个batch打印一次信息
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%')

    return train_losses, train_accuracies

# 测试模型
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():  # 测试时不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

# 训练模型（10个epochs）
train_losses, train_accuracies = train(model, train_loader, criterion, optimizer, epochs=10)

# 在测试集上评估
test_loss, test_acc = test(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'lenet5_mnist.pth')
print("模型已保存为 'lenet5_mnist.pth'")

# 绘制训练过程中的损失和准确率曲线
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy')
plt.axhline(y=test_acc, color='r', linestyle='--', label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# 可视化一些预测结果
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 转换为numpy用于显示
    images = images.cpu().numpy()

    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Pred: {predicted[i].item()}\nTrue: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 显示一些预测结果
visualize_predictions(model, test_loader)
