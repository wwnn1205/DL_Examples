import torch

# 创建张量
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 基本运算
c = a + b
d = a * 2
e = torch.dot(a, b)  # 点积

# GPU支持检测
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 张量转移
if device == "cuda":
    a_gpu = a.to(device)
    print(a_gpu.device)

print("加法结果:", c)
print("点积结果:", e.item())  # .item()获取标量值