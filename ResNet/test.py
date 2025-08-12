import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ===== 数据预处理 =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 传统 CNN =====
class PlainCNN(nn.Module):
    def __init__(self):
        super(PlainCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===== ResNet-like CNN =====
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            identity = self.shortcut(identity)
        out += identity
        return F.relu(out)

class SmallResNet(nn.Module):
    def __init__(self):
        super(SmallResNet, self).__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(1, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===== 梯度记录工具 =====
def register_hooks(model, grad_list):
    def hook_fn(module, grad_in, grad_out):
        grad_norm = grad_out[0].norm().item()
        grad_list.append(grad_norm)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_backward_hook(hook_fn)

# ===== 梯度可视化函数 =====
def visualize_gradients(model, title):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    grad_list = []
    register_hooks(model, grad_list)

    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    plt.plot(range(1, len(grad_list)+1), grad_list, marker='o')
    plt.title(title)
    plt.xlabel("Conv Layer Depth")
    plt.ylabel("Gradient L2 Norm")
    plt.show()

# ===== 对比实验 =====
visualize_gradients(PlainCNN(), "Plain CNN 梯度衰减")
visualize_gradients(SmallResNet(), "ResNet-like CNN 梯度衰减")
