
# FLIR 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸活体检测](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectLivingFace&spm=a2cio.27993362)、[红外人脸活体检测](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectInfraredLivingFace&spm=a2cio.27993362)、[视频活体检测](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectVideoLivingFace&spm=a2cio.27993362)。

IR 人脸活体检测模型FLIR


## 模型描述

用来检测图片中的人脸是否为来自认证设备端的近距离裸拍活体人脸对象，可广泛应用在人脸实时采集场景，满足人脸注册认证的真实性和安全性要求，活体判断的前置条件是图像中有人脸。



## 模型效果
![FLIR模型效果](result.png)

## 模型使用方式和使用范围
本模型可以用来判断图片中的人为真人或者是假体，分数越高则假体的可能性越高。

### 使用方式
- 推理：输入图片，如存在人脸则返回其为假体的可能性。

### 目标场景
活体模型使用场景为认证设备端和裸拍活体：
1.）认证设备端是指借助近距离裸拍活体正面人脸用于认证、通行等服务场景的含IR摄像头的硬件设备，常见的认证设备端有手机、门禁机、考勤机、PC等智能终端认证设备。
2.）裸拍活体正面人脸是指真人未经重度PS、风格化、人工合成等后处理的含正面人脸（非模糊、遮挡、大角度的正面人脸）的裸照片。常见的非真人有纸张人脸、电子屏人脸等；常见经过重度PS后处理的照片有摆拍街景照、摆拍人物风景照、摆拍证件照等；常见的其他后处理及生成照片有动漫人脸、绘画人脸等。

### 模型局限性及可能偏差
- 3D头模的拦截率一般
- 强光下对通过率和拦截率有影响

### 预处理
测试时主要的预处理如下：
根据人脸检测框信息，上下左右各扩展96./112.，遇到图像边缘则停止；把拓展后的图片的短边再对称扩展到和长边一致，遇到图像边缘则停止；若还不是正方形则再把短边对称补127到正方形，然后缩放到128x128，再centercrop出112x112进入活体模型。



### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

face_liveness_ir = pipeline(Tasks.face_liveness, 'damo/cv_manual_face-liveness_flir')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_liveness_ir.jpg'
result = face_liveness_ir(img_path)
print(f'face liveness output: {result}.')
```

### 模型训练流程
- 在私有数据集上使用SGD优化器，Warmup 5个epoch, 初始学习率为1e-2，共训练20个epoch。

## 数据评估及结果
在自建评测集上拦截/通过率为：99.8% / 97.6%

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
本模型及代码来自自研技术。

