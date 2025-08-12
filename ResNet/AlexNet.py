import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ===== 数据预处理 (CIFAR10 小图，适配 AlexNet) =====
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 原版 AlexNet（PlainCNN） =====
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ===== 残差块 =====
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            identity = self.shortcut(identity)
        out += identity
        return F.relu(out)

# ===== ResNet版 AlexNet =====
class ResNetAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetAlexNet, self).__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 64, stride=4),
            nn.MaxPool2d(3, 2),
            ResidualBlock(64, 192),
            nn.MaxPool2d(3, 2),
            ResidualBlock(192, 384),
            ResidualBlock(384, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(3, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
def get_gradients(model):
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
    return grad_list

# ===== 对比实验 =====
plain_grads = get_gradients(AlexNet())
resnet_grads = get_gradients(ResNetAlexNet())

plt.plot(range(1, len(plain_grads)+1), plain_grads, marker='o', label='Plain AlexNet')
plt.plot(range(1, len(resnet_grads)+1), resnet_grads, marker='o', label='ResNet-AlexNet')
plt.xlabel("Conv Layer Depth")
plt.ylabel("Gradient L2 Norm")
plt.title("梯度对比：Plain AlexNet vs ResNet-AlexNet")
plt.legend()
plt.show()
