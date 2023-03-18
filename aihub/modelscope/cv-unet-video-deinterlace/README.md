
# 视频去场纹模型
视频去场纹模型是一种解决隔行扫描导致的场纹问题的技术，它能够自适应地检测输入视频中的场纹，并针对场纹区域进行场纹去除与画面补全，返回画面干净自然的视频结果。

## 效果展示
以下是在真实网络视频上使用去场纹模型的结果。
<div align=center>
<img src="./data/deinterlace-example.png"  height=1080 width=960 alt="result">
</div>


## 模型描述
许多早期视频的制作过程中都使用了隔行扫描技术，这些视频每一帧都是由不同的两帧内容逐行交错得来，而两帧的不同内容会导致行间内容不对齐，从而使得画面出现场纹。此外，在视频的制作、传播过程中还会额外引入噪声、压缩失真等问题。然而，现有的很多去场纹算法都只能处理简单场景下的场纹，但对于真实且复杂场景中场纹处理效果不佳。因此，我们提出了一种新的视频去场纹模型，该模型由两个模块组成，分别是在频域上进行的去场纹模块，以及充分利用相邻帧信息以恢复画面的重建模块。我们的模型在复杂场景的视频上的去场纹效果明显优于现有其他SOTA算法。


## 期望模型使用方式以及适用范围
使用方式：
- 输入视频，模型进行拆帧、场纹去除后将输出结果。

使用范围:
- 本模型适用于一般视频去除场纹，当前只支持视频格式输入。

目标场景:
- 使用了隔行扫描技术、需要去除场纹的视频。

运行环境：
- 模型只支持GPU上运行，已在v100上测试通过。
- GPU显存要求大于等于8G。


### 如何使用
在ModelScope框架上，提供输入视频，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_deinterlace_test.mp4'
video_deinterlace_pipeline = pipeline(
    Tasks.video_deinterlace,
    'damo/cv_unet_video-deinterlace')
result = video_deinterlace_pipeline(video)[OutputKeys.OUTPUT_VIDEO]
```


### 数据预处理

- 视频拆分为图像帧。
- 对图像帧进行场纹去除。
- 使用相邻图像帧进行画面增强。

### 模型局限性以及可能的偏差

- 模型对于大部分真实场景效果良好，对于小部分降质十分严重且运动幅度大的情况可能表现不佳。

- 当前版本在python 3.7, pytorch 1.13.1环境测试通过，其他环境下可用性待测试。

## 训练数据介绍
训练数据为Youku-VESR Dataset。

### 相关论文以及引用信息
该模型借鉴了以下论文的思路或代码：
```
@article{zhao2021rethinking,
  title={Rethinking deinterlacing for early interlaced videos},
  author={Zhao, Yang and Jia, Wei and Wang, Ronggang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={7},
  pages={4872--4878},
  year={2021},
  publisher={IEEE}
}
@article{zhou2022deep,
  title={Deep Fourier Up-Sampling},
  author={Zhou, Man and Yu, Hu and Huang, Jie and Zhao, Feng and Gu, Jinwei and Loy, Chen Change and Meng, Deyu and Li, Chongyi},
  journal={arXiv preprint arXiv:2210.05171},
  year={2022}
}
```