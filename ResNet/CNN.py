import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# =======================
# 1. 数据加载与预处理
# =======================
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # 统一尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# =======================
# 2. 定义网络结构
# =======================

# 基础 CNN
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

# ResNet 示例
class ResNetLike(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetLike, self).__init__()
        self.layer1 = ResidualBlock(3, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

# =======================
# 3. 训练与测试
# =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNetLike().to(device)   # 换成 BasicCNN() 可测试基础模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[{epoch+1}] loss: {running_loss/len(trainloader):.4f}")

def test_model():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")

train_model(epochs=3)
test_model()

# =======================
# 4. 梯度可视化（梯度消失/爆炸检测）
# =======================
def visualize_gradients():
    grad_norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    plt.plot(grad_norms, marker='o')
    plt.xlabel("Layer Index")
    plt.ylabel("Gradient L2 Norm")
    plt.title("Gradient Flow")
    plt.show()

# =======================
# 5. 特征图可视化
# =======================
def visualize_feature_maps():
    dataiter = iter(testloader)
    images, _ = next(dataiter)
    images = images.to(device)
    with torch.no_grad():
        features = model.layer1(images)  # 可改成不同层
    features = features.cpu()
    fig, axarr = plt.subplots(4, 8, figsize=(12,6))
    for i in range(32):
        axarr[i//8, i%8].imshow(features[0, i].numpy(), cmap='gray')
        axarr[i//8, i%8].axis('off')
    plt.show()

# =======================
# 6. Grad-CAM（重点可解释性）
# =======================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, target_class):
        self.model.eval()
        output = self.model(input_image)
        loss = output[:, target_class]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam


# =======================
# 7. 梯度对比实验
# =======================

def train_and_record_gradients(model, loader, device):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    grad_norms_per_layer = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        layer_grad_norms = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                layer_grad_norms.append(param.grad.norm().item())
        grad_norms_per_layer.append(layer_grad_norms)
        optimizer.step()
        break  # 只跑一批数据做对比，不做完整训练

    # 取平均梯度
    grad_norms_per_layer = np.mean(grad_norms_per_layer, axis=0)
    return grad_norms_per_layer

# 创建模型并记录梯度
basic_grad_norms = train_and_record_gradients(BasicCNN(), trainloader, device)
resnet_grad_norms = train_and_record_gradients(ResNetLike(), trainloader, device)

# 绘制对比图
plt.figure(figsize=(8,5))
plt.plot(basic_grad_norms, marker='o', label='BasicCNN')
plt.plot(resnet_grad_norms, marker='o', label='ResNetLike')
plt.xlabel("Layer Index")
plt.ylabel("Gradient L2 Norm")
plt.title("Gradient Flow Comparison")
plt.legend()
plt.show()
