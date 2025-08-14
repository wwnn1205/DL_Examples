import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import random

# 1. 数据预处理
transform = transforms.Compose([
    transforms.Resize(227),  # AlexNet 需要更大的输入
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 2. 定义 AlexNet（适配 CIFAR-10）
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# 3. 测试函数
def test_model(model, testloader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 4. 训练函数（加 TensorBoard 可视化）
def train_model(model, trainloader, testloader, criterion, optimizer, epochs=5, device="cpu"):
    writer = SummaryWriter(log_dir="./runs/alexnet_cifar10")
    model.to(device)

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

        avg_loss = running_loss / len(trainloader)
        acc = test_model(model, testloader, device)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")

        # 记录 loss 和 acc
        writer.add_scalar("Loss/train", avg_loss, epoch+1)
        writer.add_scalar("Accuracy/test", acc, epoch+1)

        # 记录第一层卷积核（6 个）
        conv1_weights = model.features[0].weight.data.clone().cpu()
        conv1_weights = (conv1_weights - conv1_weights.min()) / (conv1_weights.max() - conv1_weights.min())  # 归一化
        writer.add_images("Conv1/filters", conv1_weights[:6], epoch+1)

        # 随机取一张测试图片
        sample_img, _ = random.choice(testloader.dataset)
        sample_img = sample_img.unsqueeze(0).to(device)

        # 获取第一层卷积输出
        with torch.no_grad():
            conv1_output = model.features[0](sample_img)
            conv1_output = (conv1_output - conv1_output.min()) / (conv1_output.max() - conv1_output.min())  # 归一化

        # 原图反归一化显示
        writer.add_image("Sample/original", (sample_img[0].cpu() + 1) / 2, epoch+1)

        # 取前 6 个特征图并扩展为 3 通道显示
        conv1_out_vis = conv1_output[0, :6].cpu().unsqueeze(1).repeat(1, 3, 1, 1)  # [6, 3, H, W]
        writer.add_images("Sample/conv1_output", conv1_out_vis, epoch+1)

    writer.close()

# 5. 运行
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlexNet(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_model(net, trainloader, testloader, criterion, optimizer, epochs=5, device=device)
