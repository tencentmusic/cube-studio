# pip install torch -i https://mirrors.aliyun.com/pypi/simple

import torch

# 检查是否有可用的GPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 定义模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = MyModel().to(device)

# 定义输入数据和目标标签
inputs = torch.randn(100, 10).to(device)
targets = torch.randn(100, 1).to(device)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000000):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))