import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import itertools

# =======================
# 1. 超参数
# =======================
batch_size = 64
learning_rate = 0.001
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# =======================
# 2. 数据加载
# =======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# =======================
# 3. CNN 模型
# =======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [32,16,16]
        x = self.pool(F.relu(self.conv2(x)))  # [64,8,8]
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# =======================
# 4. 损失与优化器
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =======================
# 5. TensorBoard 初始化
# =======================
writer = SummaryWriter("runs/cifar10_experiment")

# 添加模型结构图
example_data, _ = next(iter(train_loader))
writer.add_graph(model, example_data.to(device))

# =======================
# 6. 训练与验证
# =======================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.*correct/total
    avg_loss = running_loss/len(train_loader)

    # 测试集评估
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    test_acc = 100.*correct_test/total_test

    # 写入 TensorBoard
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {avg_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Test Acc: {test_acc:.2f}%")

# =======================
# 7. 卷积特征图可视化
# =======================
def visualize_feature_maps(model, images, layer_name, writer, step):
    """可视化指定卷积层的特征图"""
    activation = {}
    def hook(model, input, output):
        activation[layer_name] = output.detach()

    # 注册钩子
    layer = dict([*model.named_modules()])[layer_name]
    hook_handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        model(images.to(device))

    act = activation[layer_name].cpu()
    num_feat = min(act.shape[1], 8)  # 取前8个通道
    img_grid = torchvision.utils.make_grid(act[0, :num_feat].unsqueeze(1), normalize=True, scale_each=True)
    writer.add_image(f"FeatureMaps/{layer_name}", img_grid, step)

    hook_handle.remove()

# 取一张测试图片可视化
sample_img, _ = next(iter(test_loader))
visualize_feature_maps(model, sample_img, "conv1", writer, step=0)
visualize_feature_maps(model, sample_img, "conv2", writer, step=0)

# =======================
# 8. 混淆矩阵可视化
# =======================
from sklearn.metrics import confusion_matrix

all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

cm_fig = plot_confusion_matrix(cm, classes)
writer.add_figure("Confusion Matrix", cm_fig)

# =======================
# 9. 保存模型
# =======================
torch.save(model.state_dict(), "cifar10_cnn.pth")
writer.close()
print("模型已保存，TensorBoard 日志已生成！")
