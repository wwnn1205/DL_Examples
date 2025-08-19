import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ==============================
# 1. 数据集
# ==============================
class SimpleDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor([ord(c) for c in self.text[idx: idx + self.seq_length]], dtype=torch.long),
            torch.tensor([ord(c) for c in self.text[idx + 1: idx + self.seq_length + 1]], dtype=torch.long),
        )

text = "abcdefghijklmnopqrstuvwxyz " * 10  # 简单的字符序列
seq_length = 10
dataset = SimpleDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ==============================
# 2. Transformer 模型
# ==============================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, num_heads, num_layers,
                 ff_hidden_size, dropout=0.1, max_len=500):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, emb_size))
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, x, y):
        x_emb = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        y_emb = self.embedding(y) + self.positional_encoding[:, :y.size(1), :]
        output = self.transformer(x_emb, y_emb)
        logits = self.fc_out(output)
        return logits

# ==============================
# 3. 初始化
# ==============================
vocab_size = 256
emb_size = 64
num_heads = 8
num_layers = 3
ff_hidden_size = 256
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(vocab_size, emb_size, num_heads, num_layers, ff_hidden_size, dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# 4. 工具函数
# ==============================
def predict_next_chars(model, start_text, num_preds=20):
    """给定起始文本，生成后续字符"""
    model.eval()
    chars = [ord(c) for c in start_text]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(num_preds):
        with torch.no_grad():
            output = model(input_seq, input_seq)
            next_char_logits = output[:, -1, :]
            next_char = torch.argmax(next_char_logits, dim=-1).item()
        chars.append(next_char)
        input_seq = torch.tensor(chars[-seq_length:], dtype=torch.long).unsqueeze(0).to(device)
    return "".join([chr(c) for c in chars])

def plot_attention_matrix(model, src, tgt):
    """可视化自注意力权重"""
    model.eval()
    src_emb = model.embedding(src) + model.positional_encoding[:, :src.size(1), :]
    attn = model.transformer.encoder.layers[0].self_attn
    with torch.no_grad():
        _, attn_weights = attn(src_emb, src_emb, src_emb, need_weights=True, average_attn_weights=False)
    attn_matrix = attn_weights[0][0].cpu().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_matrix, cmap="Blues",
                xticklabels=[chr(c) for c in src[0].cpu().numpy()],
                yticklabels=[chr(c) for c in src[0].cpu().numpy()])
    plt.xlabel("Key / Value tokens")
    plt.ylabel("Query tokens")
    plt.title("Self-Attention Heatmap (Layer 1)")
    plt.show()

# ==============================
# 5. 训练
# ==============================
num_epochs = 100
train_loss = []

print("训练前预测：", predict_next_chars(model, "abcd", num_preds=20))  # 初始效果

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, y[:, :-1])
        loss = criterion(output.reshape(-1, vocab_size), y[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    train_loss.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print("训练后预测：", predict_next_chars(model, "abcd", num_preds=20))

# ==============================
# 6. 可视化
# ==============================
# (1) Loss 曲线
plt.plot(range(1, num_epochs + 1), train_loss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# (2) 随机样本预测 vs 真实值
x, y = dataset[50]
x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(x, y[:, :-1])
pred = torch.argmax(output, dim=-1)

print("\n=== 随机样本对比 ===")
print("输入序列: ", "".join([chr(c.item()) for c in x[0]]))
print("真实输出: ", "".join([chr(c.item()) for c in y[0]]))
print("模型预测: ", "".join([chr(c.item()) for c in pred[0]]))

# (3) 注意力热力图
sample_x, sample_y = dataset[100]
sample_x = sample_x.unsqueeze(0).to(device)
sample_y = sample_y.unsqueeze(0).to(device)
plot_attention_matrix(model, sample_x, sample_y)
