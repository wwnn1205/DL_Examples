import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# ========== 1. 数据加载 ==========
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet 需要 32×32 输入
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# ========== 3. 可视化函数 ==========
def visualize_feature_maps(model, image, layer_name, writer, step):
    """可视化卷积层输出的特征图"""
    outputs = []
    hooks = []

    def hook_fn(module, input, output):
        outputs.append(output)

    # 注册 hook
    for name, layer in model.named_modules():
        if name == layer_name:
            hooks.append(layer.register_forward_hook(hook_fn))

    # 前向传播一次
    with torch.no_grad():
        model(image.unsqueeze(0))

    #提取特征

    # 卷积层输出 [1, C, H, W]
    feature_maps = outputs[0].cpu()
    for i in range(feature_maps.size(1)):  # 每个通道一张图
        writer.add_image(f"{layer_name}_feature_map/{i}", feature_maps[0, i, :, :], step, dataformats="HW")

    # 移除 hook
    for h in hooks:
        h.remove()


def visualize_conv_weights(writer, model, step):
    """可视化卷积核"""
    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            kernels = param.cpu().detach().clone()
            min_v, max_v = kernels.min(), kernels.max()
            kernels = (kernels - min_v) / (max_v - min_v)  # 归一化到 0~1

            # 如果是多输入通道，取第一个输入通道
            kernels = kernels[:, 0:1, :, :]  # shape: [out_channels, 1, H, W]

            # 灰度转RGB
            kernels_rgb = kernels.repeat(1, 3, 1, 1)  # [out_channels, 3, H, W]

            writer.add_images(f"{name}_filters", kernels_rgb, step)



# ========== 4. 训练函数 ==========
def train_model(model, criterion, optimizer, epochs=5):
    writer = SummaryWriter("runs/LeNet_MNIST")
    step = 0

    # 可视化卷积核初始状态
    visualize_conv_weights(writer, model, step)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每 100 步记录一次 Loss
            if step % 100 == 0:
                writer.add_scalar("Loss/train", loss.item(), step)

            step += 1

        # 测试准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        writer.add_scalar("Accuracy/test", acc, epoch)

        # 可视化卷积核（每个 epoch 一次）
        visualize_conv_weights(writer, model, epoch)

        # 可视化第一层卷积特征图（用一张测试图片）
        sample_img, _ = test_dataset[0]
        sample_img = sample_img.to(device)
        visualize_feature_maps(model, sample_img, "conv1", writer, epoch)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}  Test Acc: {acc:.2f}%")

    writer.close()


# ========== 5. 主程序 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, epochs=5)
