import torch
print(torch.cuda.is_available())

import torch
print(torch.version.cuda)  # 显示PyTorch使用的CUDA版本
print(torch.cuda.is_available())  # 是否有可用的CUDA设备
print(torch.cuda.get_device_name(0))  # 显示第一块GPU名称