import torch
import torch.nn as nn

# 示例数据：一个简单的时间序列
data = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# 定义时间窗口大小
window_size = 3

# 将时间序列转换为输入数据和目标数据
inputs = []
targets = []
for i in range(len(data) - window_size):
    inputs.append(data[i:i+window_size])
    targets.append(data[i+window_size])

# 将输入数据和目标数据转换为张量
inputs = torch.tensor(inputs).float().unsqueeze(2)
targets = torch.tensor(targets).float().unsqueeze(1)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 定义模型参数
input_size = 1
hidden_size = 64
output_size = 1

# 创建模型实例
model = SimpleRNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
# 使用 Adam 优化器，并调整学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 2500

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 示例：使用模型进行预测
test_input = torch.tensor([[70, 80, 90]]).float().unsqueeze(2)
predicted_output = model(test_input)
print(f'Predicted next value: {predicted_output.item()}')