
# 集成中
<p align="center">
         &nbsp<a href="https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README.md">中文 </a>&nbsp| &nbsp<a href="https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md">英文</a>&nbsp | &nbsp<a href="https://github.com/bilibili/ailab">GitHub </a>&nbsp
</p>


# 图像超分辨率介绍

输入低分辨率图片，返回超高超分辨率后的高清晰图片。🔥Real-CUGAN🔥 是一个使用百万级动漫数据进行训练的，结构与Waifu2x兼容的通用动漫图像超分辨率模型。它支持2x\3x\4x倍超分辨率，其中2倍模型支持4种降噪强度与保守修复，3倍/4倍模型支持2种降噪强度与保守修复。

## 模型描述

介绍该模型的基础信息、模型特征、模型架构等。


## 期望模型使用方式以及适用范围

本模型适用范围较广，给定任意的低分辨率图片，都能生成一张高达4倍超分辨率后的高清晰度图片。

### 如何使用

在ModelScope框架上，提供低分辨图片，即可以通过简单的Pipeline调用来使用图像超分辨率模型。

#### 代码范例
`导入ms_wrapper注册脚本 (注，若要尝试其它模型，可在configuration.json中修改模型初始化参数)`
```python
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download('bilibili/cv_bilibili_image-super-resolution', revision='v0.1')
import sys
sys.path.insert(0, model_path)
import ms_wrapper
from modelscope.pipelines import pipeline
file_path = f"{model_path}/demos/title-compare1.png"
weight_path = f"{model_path}/weights_v3/up2x-latest-denoise3x.pth"
inference = pipeline('image-super-resolution', model='bilibili/cv_bilibili_image-super-resolution', weight_path=weight_path, half=False) # GPU环境可以设置为True
output = inference(file_path,tile_mode=0,cache_mode=1,alpha=1)

import matplotlib.pyplot as plt
plt.imshow(output)

```

### 模型局限性以及可能的偏差

介绍模型适用的场景，以及在哪些场景可能存在局限性，以及模型在构造训练过程中， 本身可能带有的，由于训练数据以及训练方法等因素引入的偏向性。


## 数据评估及结果
 
- **效果图对比** (推荐点开大图在原图分辨率下对比)
  <br>
  纹理挑战型(注意地板纹理涂抹)(图源:《侦探已死》第一集10分20秒)
  
  ![compare1](demos/title-compare1.png)
  
  线条挑战型(注意线条中心与边缘的虚实)(《东之伊甸》第四集7分30秒)
  
  ![compare2](demos/compare2.png)
  
  极致渣清型(注意画风保留、杂线、线条)(图源:Real-ESRGAN官方测试样例)
  
  ![compare3](demos/compare3.png)
  
  景深虚化型(蜡烛为后景，刻意加入了虚化特效，应该尽量保留原始版本不经过处理)(图源:《～闘志の華～戦国乙女2ボナ楽曲PV》第16秒)
  
  ![compare4](demos/compare4.png)
  
  <br>
- **详细对比**

|                | Waifu2x(CUNet)                                               | Real-ESRGAN(Anime6B)                                         | Real-CUGAN                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 训练集         | 私有二次元训练集，量级与质量未知                             | 私有二次元训练集，量级与质量未知                             | 百万级高清二次元patch dataset                                |
| 推理耗时(1080P)    | Baseline                                                     | 2.2x                                                         | 1x                                                           |
| 效果(见对比图) | 无法去模糊，artifact去除不干净                               | 锐化强度最大，容易改变画风，线条可能错判，<br />虚化区域可能强行清晰化 | 更锐利的线条，更好的纹理保留，虚化区域保留                   |
| 兼容性         | 大量windows-APP使用，VapourSynth支持，<br />Caffe支持，PyTorch支持，NCNN支持 | PyTorch支持，VapourSynth支持，NCNN支持                       | 同Waifu2x，结构相同，参数不同，与Waifu2x无缝兼容             |
| 强度调整       | 仅支持多种降噪强度                                           | 不支持                                                       | 已完成4种降噪程度版本和保守版，未来将支持调节不同去模糊、<br />去JPEG伪影、锐化、降噪强度 |
| 尺度           | 仅支持1倍和2倍                                               | 仅支持4倍                                                    | 已支持2倍、3倍、4倍，1倍训练中              |



### 相关论文以及引用信息

如果本模型有相关论文发表，或者是基于某些论文的结果，可以在这里 提供Bibtex格式的参考文献。
