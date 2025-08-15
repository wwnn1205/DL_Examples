import numpy as np

# 假设 sigmoid 函数和 LSTMCell 类已定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack((h_prev, x))
        gates = np.dot(self.W, combined) + self.b

        f_gate = sigmoid(gates[:hidden_size])
        i_gate = sigmoid(gates[hidden_size:2 * hidden_size])
        o_gate = sigmoid(gates[2 * hidden_size:3 * hidden_size])
        c_candidate = np.tanh(gates[3 * hidden_size:])

        c_next = f_gate * c_prev + i_gate * c_candidate
        h_next = o_gate * np.tanh(c_next)

        return h_next, c_next

# --- 模型参数和数据预处理（简化） ---

# 假设这些参数已经定义好
vocab_size = 10000       # 词汇表大小
embedding_dim = 128      # 词嵌入维度
hidden_size = 256        # LSTM 隐藏状态维度
max_sequence_length = 50 # 序列最大长度

# 假设 word_embeddings 是一个预训练好的词嵌入矩阵
# shape: (vocab_size, embedding_dim)
word_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1

# 假设这是一个情感分类数据集（简化）
# input_sequences: shape (num_samples, max_sequence_length)
# labels: shape (num_samples, 1)
num_samples = 100
input_sequences = np.random.randint(0, vocab_size, size=(num_samples, max_sequence_length))
labels = np.random.randint(0, 2, size=(num_samples, 1))

# --- 构建完整的 LSTM 分类器 ---

# LSTM 单元
lstm = LSTMCell(embedding_dim, hidden_size)

# 分类器（全连接层）参数
Why = np.random.randn(1, hidden_size) * 0.01  # 隐藏状态到输出的权重
by = np.zeros((1, 1))                         # 偏置项

# 前向传播过程（以单个样本为例）
def forward_pass(sequence_indices):
    # 初始化隐藏状态和记忆状态
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))

    # 遍历序列中的每个单词
    for word_index in sequence_indices:
        # 获取单词的词嵌入向量
        x = word_embeddings[word_index].reshape(-1, 1)

        # 传入 LSTM 单元
        h_prev, c_prev = lstm.forward(x, h_prev, c_prev)

    # 将最后一个隐藏状态作为分类器的输入
    final_h = h_prev

    # 全连接层和 Sigmoid 激活函数
    output_score = np.dot(Why, final_h) + by
    prediction = sigmoid(output_score)

    return prediction

# 示例：对第一个评论进行预测
sample_sequence = input_sequences[0]
predicted_sentiment = forward_pass(sample_sequence)

print(f"Predicted sentiment probability: {predicted_sentiment[0, 0]:.4f}")
print("Note: This code only shows the forward pass and does not include training.")