
# 天空替换模型介绍
换天算法是计算机视觉的经典任务之一，也是image sky matting的应用之一。输入一张原图A以及一张参考图B，换天算法会得到两张图片对应的天空区域的alpha图（alpha属于软分割，与硬分割mask不同，mask将视频分为前景与背景，取值为0或1，而alpha的取值范围是0到1之间，数值代表透明度，因此alpha比mask更为精细）
换天算法利用A和B图对应的天空区域alpha图，配合融合算法，将参考图B的天空样式替换到原图A中，从而改变原图A的天空样式，实现换天功能。

<div align=center>
<img src="resource/show3.png">
</div>
<br />
<br />
<div align=center>
<img src="resource/show4.png">
</div>

## 抠图系列模型

| [<img src="resource/human.png" width="280px">](https://modelscope.cn/models/damo/cv_unet_image-matting/summary) | [<img src="resource/universal.png" width="280px">](https://modelscope.cn/models/damo/cv_unet_universal-matting/summary) | [<img src="resource/video.png" width="280px">](https://modelscope.cn/models/damo/cv_effnetv2_video-human-matting/summary) |[<img src="resource/sky.png" width="280px">](https://modelscope.cn/models/damo/cv_hrnetocr_skychange/summary)|
|:--:|:--:|:--:|:--:| 
| [图像人像抠图](https://modelscope.cn/models/damo/cv_unet_image-matting/summary) | [通用抠图(支持商品、动物、植物、汽车等抠图)](https://modelscope.cn/models/damo/cv_unet_universal-matting/summary) | [视频人像抠图](https://modelscope.cn/models/damo/cv_effnetv2_video-human-matting/summary) | [天空替换(一键实现魔法换天空)](https://modelscope.cn/models/damo/cv_hrnetocr_skychange/summary) |

## 模型结构介绍
该模型由三大部分构成：低分辨率处理模块，高分辨率处理模块和换天模块；

其中低分辨率处理模块的backbone是基于hrnet-ocr框架实现的，采用了w18v2的版本，为了实现更好的效果，我们对原网络模块进行了一定修改，主要添加了自设计的super模块以及ASPP模块，扩大了感受野，重新进行分割训练，这部分的结果作为高分辨率处理模块的输入；
高分辨率处理模块基于UNET，实现对低分辨率后超分至原有分辨率，该层具有可学习参数，效果远优于直接上采样效果;
换天模块基于Multiband blender 图像拼缝技术实现；

## 使用方式和范围

使用方式：
- 支持GPU/CPU推理，在任意两张包含天空的图片上进行直接推理和换天操作。

使用范围:
- 包含天空区域的图片（3通道RGB图像，支持PNG、JPG、JPEG格式），图像分辨率建议小于5000×5000，低质图像建议预先增强处理。

目标场景:
- 艺术创作、社交娱乐。

### 如何使用

在ModelScope框架上，提供原A和参考图B，算法会将参考图B的天空样式替换到原A中，可通过简单的Pipeline调用来使用换天功能。

#### 代码范例
```python
import cv2
import os.path as osp
import modelscope
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks

image_skychange = pipeline(Tasks.image_skychange, 
                       model='damo/cv_hrnetocr_skychange')
result = image_skychange({'sky_image':'data/test/images/sky_image.jpg',
                                   'scene_image':'data/test/images/scene_image.jpg'})
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
print(f'Output written to {osp.abspath("result.png")}')
```
- sky\_image 为输入参考图B的图片
- scene\_image 为输原图A的图片

正常情况下,算法result返回的是包含换天结果的字典，可通过result[OutputKeys.OUTPUT_IMG]得到换天结果，并结合cv2.imwrite函数保存成图片，'result.png'可以替换成希望保存图片的地址和名称。

## 模型推理流程

### 图像预处理
- 图片A和图片B分别进行相关的图像预处理

### 天空区域推理
- 图片A和图片B分别进行天空区域alpha推理
- 为提升天空区域对应的alpha的高分辨率效果,alpha的尺寸和原图保存一致

### 换天
- 选取B图中有效天空区域，将参考图B的有效天空区域替换到原图A中

### 训练数据集介绍
该模型训练可以同时使用包含天空区域的语义分割数据集和matting数据集，或者使用自制数据集。



### 模型局限性以及可能的偏差
受限于训练数据集，有可能产生一些偏差，请用户自行评测后决定如何使用。
