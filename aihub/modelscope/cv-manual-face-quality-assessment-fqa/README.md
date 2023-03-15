
# FQA人脸质量评估模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸属性识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizeFace&spm=a2cio.27993362)、[表情识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizeExpression&spm=a2cio.27993362)。

FQA人脸质量评估模型


## 模型描述
FQA模型包含3个方面的创新, rank映射, Ordinal Regression和multi-labelloss:
- Rank映射：建立质量等级
- 有序回归：考虑不同rank间的排序关系
- Multi-lable loss: 解决数据不平衡问题、增强模块可扩展性


## 模型结构
![模型结构](arch.jpg)

## 质量等级划分
![模型结构](quality_rank.jpg)

## 模型效果
![模型效果](result.png)

## 模型使用方式和使用范围
本模型可以评估输入图片中人脸的质量

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

face_quality_assessment_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa')
img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png'
face_quality_score = face_quality_assessment_func(img)[OutputKeys.SCORES]
print(f'Face quality score={face_quality_score}')
```

### 使用方式
- 推理：输入经过对齐的人脸图片(112x112)，返回人脸的质量分(0-1, 分数越高质量越高)，为便于体验，集成了人脸检测和关键点模型RetinaFace。


### 目标场景
- 人脸系统基础模块，可用于人像美颜/互动娱乐/人脸比对等场景.


### 模型局限性及可能偏差
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 模型性能指标
![模型性能曲线](performance.jpg)

## 来源说明
本模型及代码来自达摩院自研技术
