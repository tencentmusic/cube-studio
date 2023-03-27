
# Fer 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸属性识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizeFace&spm=a2cio.27993362)、[表情识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizeExpression&spm=a2cio.27993362)。

人脸表情识别模型Fer


## 模型描述

Fer为人脸表情识别领域的明星项目([代码地址](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch))。网络结构比较简单，backbone是VGG19, 后面接一个fc层。
VGG19是Oxford的Visual Geometry Group的组提出的,该网络是在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。VGG19相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。


VGG19:

![模型结构](https://modelscope.cn/api/v1/models/damo/cv_vgg19_facial-expression-recognition_fer/repo?Revision=master&FilePath=vgg_19.jpg&View=true)

Fer Demo：

![模型效果](https://modelscope.cn/api/v1/models/damo/cv_vgg19_facial-expression-recognition_fer/repo?Revision=master&FilePath=fer.jpg&View=true)


## 模型使用方式和使用范围
本模型可以输出图像中人脸的表情

### 使用方式
- 推理：输入带人脸的图片，得到对应的表情，包括生气，厌恶，害怕，高兴，悲伤，惊讶，中立。

### 目标场景
- 人脸相关的基础能力，可应用于人脸分析等场景。

### 模型局限性及可能偏差
- 多人脸表情识别暂不支持
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 预处理
测试时主要的预处理如下：
- Normalize：图像归一化，减均值除以标准差

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
import numpy as np

fer = pipeline(Tasks.facial_expression_recognition, 'damo/cv_vgg19_facial-expression-recognition_fer')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facial_expression_recognition.jpg'
ret = fer(img_path)
label_idx = np.array(ret['scores']).argmax()
label = ret['labels'][label_idx]
print(f'facial expression : {label}.')

```

### 模型训练流程
- 在FER2013数据集上使用SGD优化器，初始学习率为1e-2，batch size 为128，训练250个epoch。

### 测试集介绍
- CK+ 数据集：CK+数据集是CK数据集的扩展。它包含327个标记好的面部视频。我们从CK+数据集中的每个序列中提取了最后三帧，总共包含981个面部表情。我们在实验中使用了10折交叉验证。
- fer2013 数据集：fer2013测试集包含3589个样本，来自[kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)。

## 数据评估及结果
模型在WiderFace的验证集上客观指标如下：
| Method | fer2013 | CK+ |
| ------------ | ------------ | ------------ | 
| Fer | 71.496% | 94.646% | 
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
本模型及代码来自开源社区([地址](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch))，请遵守相关许可。
