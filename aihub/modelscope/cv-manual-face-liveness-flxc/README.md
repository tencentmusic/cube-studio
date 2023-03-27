
# FLXC 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸活体检测](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectLivingFace&spm=a2cio.27993362)、[红外人脸活体检测](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectInfraredLivingFace&spm=a2cio.27993362)、[视频活体检测](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectVideoLivingFace&spm=a2cio.27993362)。

静默炫彩人脸活体检测模型FLXC


## 模型描述

用来检测图片中的人脸是否为来自认证设备端的近距离裸拍活体人脸对象，可广泛应用在人脸实时采集场景，满足人脸注册认证的真实性和安全性要求，活体判断的前置条件是图像中有人脸。炫彩活体检测的原理是通过光在活体和假体上的不同反射情况来对活体和假体进行区分，其特点主要有:
- 不用额外或特殊的传感器
- 对多种攻击鲁棒，特别是深度有关的攻击（如非肤质3D面具，和头模等）


## 模型原理
![FLXC模型原理](https://modelscope.cn/api/v1/models/damo/cv_manual_face-liveness_flxc/repo?Revision=master&FilePath=algo.jpg&View=true)

## 模型效果
![FLIR模型效果](result.png)

## 模型使用方式和使用范围
本模型可以用来判断图片中的人为真人或者是假体，分数越高则假体的可能性越高。

### 使用方式
- 推理：输入图片，如存在人脸则返回其为假体的可能性。

### 目标场景
活体模型使用场景为认证设备端和裸拍活体：
1.）认证设备端是指借助近距离裸拍活体正面人脸用于认证、通行等服务场景的含RGB摄像头的硬件设备，常见的认证设备端有手机、门禁机、考勤机、PC等智能终端认证设备。
2.）裸拍活体正面人脸是指真人未经重度PS、风格化、人工合成等后处理的含正面人脸（非模糊、遮挡、大角度的正面人脸）的裸照片。常见的非真人有纸张人脸、电子屏人脸等；常见经过重度PS后处理的照片有摆拍街景照、摆拍人物风景照、摆拍证件照等；常见的其他后处理及生成照片有动漫人脸、绘画人脸等。

### 模型局限性及可能偏差
- 目前支持的是静默炫彩活体检测，即只支持单图的输入，导致模型不太鲁邦，实际使用建议输入序列图片，之后vote多个模型输出的结果来进行判断。

### 预处理
测试时主要的预处理如下：
根据人脸检测框信息，上下左右各扩展96./112.，遇到图像边缘则停止；把拓展后的图片的短边再对称扩展到和长边一致，遇到图像边缘则停止；若还不是正方形则再把短边对称补127到正方形，然后缩放到128x128，再centercrop出112x112的图片feed进活体模型。



### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

face_liveness_xc = pipeline(Tasks.face_liveness, 'damo/cv_manual_face-liveness_flxc')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_liveness_rgb.png'
result = face_liveness_xc(img_path)
print(f'face liveness output: {result}.')
```

### 模型训练流程
- 在私有数据集上使用SGD优化器，Warmup 5个epoch, 初始学习率为1e-2，共训练20个epoch。

## 数据评估及结果
在自建评测集上拦截/通过率为：99.8% / 92.1%

## 来源说明
本模型及代码来自自研技术。

