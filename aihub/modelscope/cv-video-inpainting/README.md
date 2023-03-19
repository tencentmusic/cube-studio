
# 视频编辑



这是一个video inpainting模型。输入一段视频和mask区域，对视频进行修复。




## 模型描述

利用transfomer结构，基于[STTN](https://github.com/researchmm/STTN).进行了优化，适用深度可分离卷积替代标准卷积，同时对模型的宽度进行了瘦身。

## 使用方式和范围


### 如何使用

在ModelScope框架上，提供输入视频和mask目录，即可以通过简单的Pipeline调用来使用视频编辑/修复。

#### 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_inpainting = pipeline(Tasks.video_inpainting, 
                       model='damo/cv_video-inpainting')
result_status = video_inpainting({'video_input_path':'data/test/videos/video_inpainting_test.mp4',
                           'mask_path':'data/test/videos/mask_dir',
                           'video_output_path':'out.mp4'})
result = result_status[OutputKeys.OUTPUT]

```
- video\_input\_path 为输入视频的路径
- mask_path 为mask文件的目录，支持一个视频对应多个mask，mask文件的命名规则示例：mask\_00000\_000321.png，mask\_000322\_000632.png，分别代表第1帧到第322帧的mask，和第323帧到底633帧的mask。mask中需要修复的区域像素为(0, 0, 0)，其他区域像素为(255, 255, 255)
- video\_output\_path 为输出视频的路径

正常情况下，算法返回字符串‘Done’，如果遇到输入视频无法解码时，算法返回字符串‘decode_error’

### 模型局限性以及可能的偏差

- 对于文字背景，直线、曲线等规则图案的背景，修复效果会受到影响。


## 训练数据介绍

训练数据来自互联网搜索的视频

## 模型推理流程

### 预处理

- 算法会根据mask的区域，自适应对对视频进行crop，然后进行推理。

### 推理

- 会把连续的300帧送入模型，进行推理，得到修复结果


## 引用
```
@inproceedings{yan2020sttn,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang,
  title = {Learning Joint Spatial-Temporal Transformations for Video Inpainting},
  booktitle = {The Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```