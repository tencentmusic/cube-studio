# 1. 什么是Face-Paint

中文叫做：图像风格动画化

顾名思义，是可以通过算法将照片颜色以及照片风格等，调整为动漫的风格的强大算法！💪

# 2. 我用它能实现什么样的效果？

![](https://user-images.githubusercontent.com/26464535/142294796-54394a4a-a566-47a1-b9ab-4e715b901442.gif)

## 😻怎么样！想在本地运行它吗？

# 

# 3. 如何在本地运行它？

#### 1.安装pytorch

   执行以下命令，安装pytorch

```bash
pip install torch>=1.7.1 torchvision
```

或可进入[pytorch官网](https://pytorch.org/)选择所需要安装的版本安装

#### 2. 将项目文件缓存至本地

可通过此命令进行

```python
import torch
model = torch.hub.load("bryandlee/animegan2-pytorch", "generator").eval()
out = model(img_tensor)
```

#### 3. 加载项目的训练好的权重文件

可以通过代码方式加载至缓存或[点击下载](https://github.com/bryandlee/animegan2-pytorch/tree/main/weights)

在此列出通过代码加载权重至缓存的方式

```python
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="celeba_distill")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
```

#### 4. 加载基础的修改图像风格方法

执行此代码用于将主要方法动态加载到缓存中，便于调用

```python
# 参数 size 用于控制方法输出图片的长宽大小
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)
```

#### 

#### 5. 恭喜！读取图片，获得结果吧！

```python
from PIL import Image

img = Image.open("填写图片文件路径").convert("RGB")
out = face2paint(model, img) # 此处model为第三步加载的model
out.show()  # 显示结果😊
```

### 😊仍然没有运行成功？

#### 可进入[项目notebook](https://github.com/tencentmusic/cube-studio/blob/master/aihub/deep-learning/face-paint/face-paint.ipynb)查看具体调用方法，🎉内含人像动漫化及更换背景等~~
