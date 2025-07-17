import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 生成数据
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 3 + torch.randn(100, 1) * 2  # y = 2x + 3 + 噪声


# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 输入1维，输出1维

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
epochs = 100
losses = []
for epoch in range(epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 可视化
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 预测
with torch.no_grad():
    predicted = model(X).detach().numpy()
    plt.scatter(X, y, label='Original Data')
    plt.plot(X, predicted, 'r-', label='Fitted Line')
    plt.legend()
    plt.show()