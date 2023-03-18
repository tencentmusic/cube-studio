

# 图像人脸融合

给定一张模板图和一张目标用户图，图像人脸融合模型能够自动地将用户图中的人脸融合到模板人脸图像中，生成一张与目标人脸相似，且具有模版图外貌特征的新图像。

其生成效果如下所示：

![生成效果](description/demo.png)

## 模型描述

本模型使用多尺度属性编码器提取模板图属性特征，使用预训练人脸识别模型提取用户图的ID特征，再通过引入可行变特征融合结构，
将ID特征嵌入属性特征空间的同时，以光流场的形式实现面部的自适应变化，最终融合结果真实，高保真，且支持一定程度内对目标用户脸型的自适应感知。

![模型结构](description/architecture.png)

## 使用方式和范围

使用方式：
- 直接推理，在任意真实人物图像对上进行直接推理。

使用范围:
- 正脸或偏侧一定小范围角度的侧脸图像，人脸五官轮廓清晰无遮挡，图像分辨率大于128x128，小于4000×4000。

目标场景:
- 互动娱乐，广告宣传，影楼试片等。

### 如何使用

在ModelScope框架上，提供输入的模板图和用户图，即可以通过简单的Pipeline调用来使用。

#### 代码范例
```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

image_face_fusion = pipeline(Tasks.image_face_fusion, 
                       model='damo/cv_unet-image-face-fusion_damo')
template_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_template.jpg'
user_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_user.jpg'
result = image_face_fusion(dict(template=template_path, user=user_path))

cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
print('finished!')

```

### 模型局限性以及可能的偏差

- 建议图像中人脸五官区域轮廓完整，无明显遮挡，否则可能影响人脸检测结果导致融合效果不佳。
  
- 算法支持人脸偏侧一定角度，在偏侧角度不超过30度的情况下能取得更佳效果。

- 图像中人脸尺寸建议大于64×64像素，人脸区域建议不大于全图区域的2/3，否则会影响人脸检测的结果导致无法进行融合操作。

- 如果两张人脸的脸型差别过大，可能影响脸部边缘区域的融合效果。

## 模型推理流程

### 预处理

- 人脸区域提取&对齐，得到256x256大小的图像用于模型推理。
- 人脸识别模型提取用户人脸的脸部特征。
- 3d重建网络对模板图重建三维结构信息。
  


