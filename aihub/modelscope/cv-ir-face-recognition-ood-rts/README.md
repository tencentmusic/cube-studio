
# RTS模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFace&spm=a2cio.27993362)、[口罩人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFaceWithMask&spm=a2cio.27993362)、[人脸搜索1:N](https://vision.aliyun.com/experience/detail?tagName=facebody&children=SearchFace&spm=a2cio.27993362)、[公众人物识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizePublicFace&spm=a2cio.27993362)、[明星识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectCelebrity&spm=a2cio.27993362)。

人脸识别OOD模型RTS


## 模型描述
针对人脸识别系统中经常遇到的低质量、噪声、甚至不同数据分布（out of distribution， OOD）的数据带来的问题，用基于概率的视角分析损失函数中温度调节参数和分类不确定度的内在关系，同时该不确定度服从一个先验分布。从而可以稳定训练，以及在部署时提供一个对不确定度的度量分值，帮助建立更鲁棒的人脸识别系统。
主要贡献点如下：
（1）基于概率视角，揭示了损失函数中温度调节参数和分类不确定度的内在关系，通过提出的Random Temperature Scaling (RTS) 来训练更可靠的人脸识别模型。
（2）在训练阶段，RTS可以调节干净数据和噪声数据对训练的影响以得到更稳定的训练过程和更好的识别效果。
（3）在测试阶段，RTS可以提供一个不需要通过额外数据训练的不确定度分值，来分辨出不确定的、低质量的以及不同数据分布（out of distribution， OOD）的样本，以建立更鲁棒的人脸识别系统。


## 模型效果
![模型效果](rts.png)

## 模型使用方式和使用范围
本模型可以检测输入图片中人脸的512维特征和对应的质量分。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

rts_face_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_ir_face-recognition-ood_rts')

img1 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png'
img2 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_2.png'

result1 = rts_face_recognition_func(img1) 
result2 = rts_face_recognition_func(img2) 

emb1 = result1[OutputKeys.IMG_EMBEDDING]
score1 = result1[OutputKeys.SCORES][0][0]

emb2 = result2[OutputKeys.IMG_EMBEDDING]
score2 = result2[OutputKeys.SCORES][0][0]

sim = np.dot(emb1[0], emb2[0])
print(f'Cos similarity={sim:.3f}, img1:{img1}  img2:{img2}') 
print(f'OOD score: img1:{score1:.3f}  img2:{score2:.3f}') 

```

### 使用方式
- 推理：输入经过对齐的人脸图片(112x112)，返回人脸特征向量(512维)，为便于体验，集成了人脸检测和关键点模型RetinaFace，输入两张图片，各自进行人脸检测选择最大脸并对齐后提取特征，然后返回相似度比分以及每个人脸的质量分。

### 目标场景
- 人脸识别应用广泛，如考勤，通行，人身核验，智慧安防等场景

### 模型局限性及可能偏差
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 预处理
测试时主要的预处理如下：
- Resize：图像resize到112x112
- Normalize：图像归一化，减均值除以标准差

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
本模型及代码来自达摩院自研技术

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
```

