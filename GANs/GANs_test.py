import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 准备数据 ---
# 与 VAE 代码一样，加载并准备 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# --- 2. 定义判别器 (Discriminator) ---
# 判别器的任务是判断一张图是真实的还是生成的
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2), # 使用 LeakyReLU 激活函数，有助于解决梯度消失问题
            nn.Linear(128, 1),
            nn.Sigmoid() # Sigmoid 将输出压缩到 0-1 之间，代表真伪概率
        )

    def forward(self, x):
        return self.model(x)

# --- 3. 定义生成器 (Generator) ---
# 生成器的任务是根据随机噪声生成假图像
class Generator(nn.Module):
    def __init__(self, z_dim=100, output_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(z_dim, 512),
        nn.ReLU(True),
        nn.Linear(512, 1024),
        nn.ReLU(True),
        nn.Linear(1024, output_dim),
        nn.Tanh()
)


    def forward(self, x):
        return self.model(x)

# --- 4. 训练模型 ---
# 设置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100 # 随机噪声的维度
lr = 0.0002
epochs = 150

# 实例化模型
discriminator = Discriminator().to(device)
generator = Generator(z_dim=z_dim).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss() # 二元交叉熵损失
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# 设置用于可视化的真实和虚假标签
real_label = 1.0
fake_label = 0.0

# 训练循环
for epoch in range(epochs):
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
        for i, (images, _) in enumerate(pbar):
            # 将图片展平并移到设备上
            images = images.view(-1, 28 * 28).to(device)
            batch_size = images.size(0)

            # --- 训练判别器 ---
            d_optimizer.zero_grad()

            # 1. 训练判别器识别“真实”图像
            real_output = discriminator(images)
            d_loss_real = criterion(real_output, torch.full_like(real_output, real_label))
            d_loss_real.backward()

            # 2. 训练判别器识别“虚假”图像
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise).detach() # .detach() 阻止梯度传到生成器
            fake_output = discriminator(fake_images)
            d_loss_fake = criterion(fake_output, torch.full_like(fake_output, fake_label))
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()

            # --- 训练生成器 ---
            g_optimizer.zero_grad()

            # 生成器希望判别器认为其生成的图像是“真实的”
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise)
            output = discriminator(fake_images)
            g_loss = criterion(output, torch.full_like(output, real_label))
            g_loss.backward()
            g_optimizer.step()

            # 更新进度条
            pbar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

print("训练完成！")

# --- 5. 生成新图片（看效果）---
generator.eval()
with torch.no_grad():
    # 从潜在空间中随机生成 16 个点
    noise = torch.randn(16, z_dim).to(device)
    # 使用生成器生成图片
    generated_images = generator(noise).cpu()

    # 恢复图片的像素范围并保存
    generated_images = (generated_images + 1) / 2 # 将 -1,1 范围转换回 0,1
    generated_images = generated_images.view(-1, 1, 28, 28)
    torchvision.utils.save_image(generated_images, 'gan_generated_digits.png', nrow=4, pad_value=1)

print("生成了 16 张新图片，已保存为 gan_generated_digits.png")
print("快去项目文件夹中查看效果吧！")