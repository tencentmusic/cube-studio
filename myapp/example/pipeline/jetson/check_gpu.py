import torch
print('torch version:',torch.__version__)          # 查看 PyTorch 版本
print('torch cuda version:',torch.version.cuda)         # 查看编译时用的 CUDA 版本
print('torch cuda is_available:',torch.cuda.is_available())  # 检查 CUDA 是否可用
print('torch cuda count:',torch.cuda.device_count())  # 直接获取 GPU 数量