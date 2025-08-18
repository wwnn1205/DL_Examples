import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. 准备数据 ---
# 自动下载并加载 MNIST 数据集
# 将图片转换为 PyTorch 张量，并标准化
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# --- 2. 定义 VAE 模型 ---
class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=256, z_dim=20):
        super(VAE, self).__init__()

        # 编码器 (Encoder): 压缩数据
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim), # 输入层到隐藏层
            nn.ReLU(),
            # 这里是 VAE 的关键：不直接输出一个点，而是输出两个值
            # 一个是均值 (mean)，另一个是标准差的对数 (log_variance)
            # log_variance 是为了保证方差为正数
        )
        self.fc_mu = nn.Linear(h_dim, z_dim) # 隐藏层到均值
        self.fc_logvar = nn.Linear(h_dim, z_dim) # 隐藏层到方差的对数

        # 解码器 (Decoder): 生成数据
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), # 潜在空间到隐藏层
            nn.ReLU(),
            nn.Linear(h_dim, input_dim), # 隐藏层到输出层
            nn.Sigmoid() # 使用 Sigmoid 激活函数，将输出值压缩到 0-1 之间，对应图片像素
        )

    def reparameterize(self, mu, log_var):
        """
        重参数化技巧 (Reparameterization Trick)
        这是 VAE 的核心，使得我们可以通过反向传播来训练模型。
        从一个正态分布中取样，然后通过 mu 和 log_var 调整。
        """
        std = torch.exp(0.5 * log_var) # 从对数方差计算标准差
        eps = torch.randn_like(std) # 从标准正态分布中随机取样一个噪声
        return mu + eps * std # 最终的潜在变量 z = 均值 + 噪声 * 标准差

    def forward(self, x):
        # 前向传播过程
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

# --- 3. 定义损失函数 ---
def vae_loss_function(x_reconstructed, x, mu, log_var):
    # 1. 重构损失 (Reconstruction Loss): 衡量生成图像和原始图像的相似度
    # 这里使用二元交叉熵损失，适用于像素值在 0-1 之间的图像
    reconstruction_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')

    # 2. KL 散度 (KL Divergence): 衡量潜在空间分布和标准正态分布的相似度
    # 这部分是 VAE 的精髓，它鼓励潜在空间分布接近正态分布，从而让模型“学会”生成新数据
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return reconstruction_loss + kl_divergence

# --- 4. 训练模型 ---
# 设置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

model.train()
for epoch in range(epochs):
    total_loss = 0
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
        for i, (images, _) in enumerate(pbar):
            # 将图片展平，从 28x28 变为 784
            images = images.view(-1, 28*28).to(device)

            # 前向传播
            reconstructed_images, mu, log_var = model(images)
            loss = vae_loss_function(reconstructed_images, images, mu, log_var)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (i + 1))

print("训练完成！")

# --- 5. 生成新图片（看效果）---
model.eval()
with torch.no_grad():
    # 从潜在空间中随机生成 16 个点
    z = torch.randn(16, 20).to(device)
    # 使用解码器生成图片
    generated_images = model.decoder(z)

    # 将生成的图片保存下来
    generated_images = generated_images.view(-1, 1, 28, 28).cpu()
    torchvision.utils.save_image(generated_images, 'generated_digits.png', nrow=4, pad_value=1)

print("生成了 16 张新图片，已保存为 generated_digits.png")
print("快去项目文件夹中查看效果吧！")