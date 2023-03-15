
# FRIR 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFace&spm=a2cio.27993362)、[口罩人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFaceWithMask&spm=a2cio.27993362)、[人脸搜索1:N](https://vision.aliyun.com/experience/detail?tagName=facebody&children=SearchFace&spm=a2cio.27993362)、[公众人物识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizePublicFace&spm=a2cio.27993362)、[明星识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectCelebrity&spm=a2cio.27993362)。

IR人脸识别模型FRIR



## 模型描述
FRIR是基于残差网络结构的对红外（IR）人脸提取人脸特征的模型。输入为对齐之后的112x112的人脸IR图片，输出为512维的IR人脸特征。通过比较两张人脸的cosine相似度可以判断两张图片是否为同一个人。

## 模型结构
FRIR采用残差网络结构，loss采用arcface loss。

![FRIR模型结构](arcface.jpg)

## 模型使用方式和使用范围
可用于红外成像的人脸比对，例如低成本红外摄像头的门禁或门锁场景。

### 使用方式
- 推理：输入一张带人脸的图片，经过检测器和对齐模块得到，对齐的人脸图片(112x112)，经过本模型后返回人脸特征向量(512维)，为便于体验，集成了人脸检测和关键点模型RetinaFace，输入两张图片，各自进行人脸检测选择最大脸并对齐后提取特征，然后返回相似度比分以及每个人脸的质量分。

### 目标场景
活体模型使用场景为认证设备端和裸拍活体：
1.）认证设备端是指借助近距离裸拍活体正面人脸用于认证、通行等服务场景的含RGB摄像头的硬件设备，常见的认证设备端有手机、门禁机、考勤机、PC等智能终端认证设备。
2.）裸拍活体正面人脸是指真人未经重度PS、风格化、人工合成等后处理的含正面人脸（非模糊、遮挡、大角度的正面人脸）的裸照片。常见的非真人有纸张人脸、电子屏人脸等；常见经过重度PS后处理的照片有摆拍街景照、摆拍人物风景照、摆拍证件照等；常见的其他后处理及生成照片有动漫人脸、绘画人脸等。

### 模型局限性及可能偏差
- 强光照射下的红外图片提取特征的效果一般。
- 当前版本在python 3.7, pytorch 1.8.0环境测试通过，其他环境下可用性待测试.



### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

face_recognition_frir_func = pipeline(Tasks.face_recognition, 'damo/cv_manual_face-recognition_frir')
img1 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ir_face_recognition_1.png'
img2 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ir_face_recognition_2.png'
emb1 = face_recognition_frir_func(img1)[OutputKeys.IMG_EMBEDDING]
emb2 = face_recognition_frir_func(img2)[OutputKeys.IMG_EMBEDDING]
sim = np.dot(emb1[0], emb2[0])
print(f'Face cosine similarity={sim:.3f}, img1:{img1}  img2:{img2}')
```


## 数据评估及结果
私有数据集下，100人底库，1e-5的误识率下，通过率97%。

## 来源说明
本模型及代码来自达摩院自研技术。

