
# ULFD 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸检测与五官定位](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectFace&spm=a2cio.27993362)。

超快人脸检测模型ULFD, github上高star repo。


## 模型描述

ULFD([代码地址](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB))为轻量级人脸检测算法, 基于SSD框架手工设计了backbone结构，是业界开源的第一个1M人脸检测模型。当输入320x240分辨率的图片且未使用onnxruntime加速时，在CPU上跑需要50-60ms，当使用onnxruntime加速后，在CPU上仅需要8-11ms, 优点如下:
- 在模型大小方面，默认 FP32 精度下（.pth）文件大小为 1.1MB，推理框架 int8 量化后大小为 300KB 左右。
- 在模型计算量方面，320x240 的输入分辨率下仅 90~109 MFlops 左右，足够轻量。
- 无特殊算子，支持 onnx 导出，便于移植推理

模型结构与SSD一致，backbone为作者自己设计的结构，包含13个模块，主要由depth-wise卷积组成，分类回归的计算共有4个分支，前3个分支分别以backbone的第8，11，13个模块的输出作为输入；在backbone之后跟一个附加模块，其输出接入第4个分支。

![模型结构](https://modelscope.cn/api/v1/models/damo/cv_manual_face-detection_ulfd/repo?Revision=master&FilePath=ssd.jpg&View=true)

## 模型使用方式和使用范围
本模型可以检测输入图片中人脸的位置。

### 使用方式
- 推理：输入图片，如存在人脸则返回人脸位置，可检测多张人脸


### 目标场景
- 人脸相关的基础能力，可应用于人像美颜/互动娱乐/人脸比对等场景

### 模型局限性及可能偏差
- 部分遮挡的人脸检测效果可能一般
- 由于模型较小，多人脸场景下可能出现漏检的情况
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 预处理
测试时主要的预处理如下：
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks

ulfd_face_detection = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_ulfd')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ulfd_face_detection.jpg'
result = ulfd_face_detection(img_path)
print(f'face detection output: {result}.')
```

### 模型训练流程

- 在Wider Face数据集上使用SGD优化器，初始学习率为1e-2，batch size 为24，训练了200个 epoch。

### 数据评估及结果
模型在WiderFace的验证集上客观指标如下：
| Method | Easy | Medium | Hard |
| ------------ | ------------ | ------------ | ------------ |
| ULFD | 85.3 | 81.9 | 53.9 |

### 测试集
- WIDERFACE: 测试集已上传至ModelScope的DatasetHub, 详情请见[WIDER_FACE](https://modelscope.cn/datasets/shaoxuan/WIDER_FACE)。

## 人脸相关模型

以下是ModelScope上人脸相关模型:

- 人脸检测

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [RetinaFace人脸检测模型](https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary) |
| 2 | [MogFace人脸检测模型-large](https://modelscope.cn/models/damo/cv_resnet101_face-detection_cvpr22papermogface/summary) |
| 3 | [TinyMog人脸检测器-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_tinymog/summary) |
| 4 | [ULFD人脸检测模型-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_ulfd/summary) |
| 5 | [Mtcnn人脸检测关键点模型](https://modelscope.cn/models/damo/cv_manual_face-detection_mtcnn/summary) |
| 6 | [ULFD人脸检测模型-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_ulfd/summary) |


- 人脸识别

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [口罩人脸识别模型FaceMask](https://modelscope.cn/models/damo/cv_resnet_face-recognition_facemask/summary) |
| 2 | [口罩人脸识别模型FRFM-large](https://modelscope.cn/models/damo/cv_manual_face-recognition_frfm/summary) |
| 3 | [IR人脸识别模型FRIR](https://modelscope.cn/models/damo/cv_manual_face-recognition_frir/summary) |
| 4 | [ArcFace人脸识别模型](https://modelscope.cn/models/damo/cv_ir50_face-recognition_arcface/summary) |
| 5 | [IR人脸识别模型FRIR](https://modelscope.cn/models/damo/cv_manual_face-recognition_frir/summary) |

- 人脸活体识别

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [人脸活体检测模型-IR](https://modelscope.cn/models/damo/cv_manual_face-liveness_flir/summary) |
| 2 | [人脸活体检测模型-RGB](https://modelscope.cn/models/damo/cv_manual_face-liveness_flrgb/summary) |
| 3 | [静默人脸活体检测模型-炫彩](https://modelscope.cn/models/damo/cv_manual_face-liveness_flxc/summary) |

- 人脸关键点

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [FLCM人脸关键点置信度模型](https://modelscope.cn/models/damo/cv_manual_facial-landmark-confidence_flcm/summary) |

- 人脸属性 & 表情


| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [人脸表情识别模型FER](https://modelscope.cn/models/damo/cv_vgg19_facial-expression-recognition_fer/summary) |
| 2 | [人脸属性识别模型FairFace](https://modelscope.cn/models/damo/cv_resnet34_face-attribute-recognition_fairface/summary) |


## 来源说明
本模型及代码来自开源社区([地址](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB))，请遵守相关许可。
