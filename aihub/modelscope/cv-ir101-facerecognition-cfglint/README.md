
# CurricularFace 模型介绍


## 模型描述

CurricularFace为目前人脸识别的SOTA方法之一，其主要思想是采用课程学习的思想动态关注训练数据中的难例样本。此前的方法在训练中对于错分类的样本，要么未充分挖掘难例导致性能问题，要么在训练初期关注难例导致收敛问题。基于此，CurricularFace提出Adaptive Curriculum Learning Loss, 在训练过程中动态调整easy和hard样本的重要性，使训练初期关注简单样本，训练后期关注难例样本，而对难易样本分配不同重要性是通过设计一个代表收敛进度的指示函数来自适应调整的。论文已发表至CVPR-2020（[论文地址](https://arxiv.org/abs/2004.00288), [代码地址](https://github.com/Tencent/TFace/tree/master/recognition)）。

## 训练数据

本模型的训练数据来自开源社区规模最大的数据集之一Glint360K，包含36万ID，1700万图片(详见：[论文地址](https://arxiv.org/abs/2203.15565)), 因此该模型在学界的多个benchmark中处于SOTA水平。


## 使用方式和范围

使用方式：
- 推理：输入经过对齐的人脸图片(112x112)，返回人脸特征向量(512维)，为便于体验，集成了人脸检测和关键点模型[SCRFD](https://modelscope.cn/models/damo/cv_resnet_facedetection_scrfd10gkps)，输入两张图片，各自进行人脸检测选择最大脸并对齐后提取特征，然后返回相似度比分

目标场景:
- 人脸识别应用广泛，如考勤，通行，人身核验，智慧安防等场景

#### 代码范例
```python
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img1 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png'
img2 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_2.png'

face_recognition = pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint')
emb1 = face_recognition(img1)[OutputKeys.IMG_EMBEDDING]
emb2 = face_recognition(img2)[OutputKeys.IMG_EMBEDDING]
sim = np.dot(emb1[0], emb2[0])
print(f'Face cosine similarity={sim:.3f}, img1:{img1}  img2:{img2}')
```

## 模型性能指标

在IJBC数据集中，模型在不同FPR下的TPR指标(注：使用不同的检测及关键点模型会对识别指标有细微影响):

| Name | 1e-6 | 1e-5 | 1e-4 |
| ------------ | ------------ | ------------ |------------ |
| CF_IR101_Glint | 92.01 | 96.17 | 97.47 |

## 来源说明
本模型及代码来自开源社区([地址](https://github.com/Tencent/TFace/tree/master/recognition))，请遵守相关许可。
