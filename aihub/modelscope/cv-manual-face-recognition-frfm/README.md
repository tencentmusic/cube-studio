
# FRFM模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFace&spm=a2cio.27993362)、[口罩人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFaceWithMask&spm=a2cio.27993362)、[人脸搜索1:N](https://vision.aliyun.com/experience/detail?tagName=facebody&children=SearchFace&spm=a2cio.27993362)、[公众人物识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizePublicFace&spm=a2cio.27993362)、[明星识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectCelebrity&spm=a2cio.27993362)。

口罩人脸识别模型FRFM。


## 模型描述
口罩人脸识别模型FRFM基于ArcFace框架，网络结构为NAS搜索出来的large model，训练数据使用了Glint360数据，同时加了数据增强的策略，包括random erasing，cutout策略和以一定概率增加口罩, 大大增强了在口罩数据集上的精度, 在MFR数据集中性能可以在**96以上**，同时在标准人脸识别任务上的精度也有提高。


## 模型结构
![模型结构](https://modelscope.cn/api/v1/models/damo/cv_manual_face-recognition_frfm/repo?Revision=master&FilePath=arcface.jpg&View=true)

## 模型使用方式和使用范围
本模型可以检测输入图片中带口罩和不戴口罩人脸的特征

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

face_mask_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_manual_face-recognition_frfm')
img1 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png'
img2 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_2.png'
emb1 = face_mask_recognition_func(img1)[OutputKeys.IMG_EMBEDDING]
emb2 = face_mask_recognition_func(img2)[OutputKeys.IMG_EMBEDDING]
sim = np.dot(emb1[0], emb2[0])
print(f'Face cosine similarity={sim:.3f}, img1:{img1}  img2:{img2}')
```

### 使用方式
- 推理：输入经过对齐的人脸图片(112x112)，返回人脸特征向量(512维)，为便于体验，集成了人脸检测和关键点模型RetinaFace，输入两张图片，各自进行人脸检测选择最大脸并对齐后提取特征，然后返回相似度比分


### 目标场景
- 人脸识别应用广泛，如考勤，通行，人身核验，智慧安防等场景


### 模型局限性及可能偏差
- 由于模型较大，目前仅支持GPU推理。
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 模型性能指标

| Method | IJBC(1e-5) | IJBC(1e-4) | MFR-ALL |
| ------------ | ------------ | ------------ | ------------ |
| FaceMask | - | -  | 96+ |

## 来源说明
本模型及代码来自达摩院和Insightface联合研发技术
