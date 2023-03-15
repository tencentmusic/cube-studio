
# TinyMog 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸检测与五官定位](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectFace&spm=a2cio.27993362)。
轻量级人脸检测模型TinyMog

## 模型描述
本模型基于Nas搜索出来的backone + scrfd-0.5G的head module 和FPN结构。

## 模型使用方式和使用范围
本模型可以检测输入图片中人脸的位置。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

tinymog_face_detection_func = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_tinymog')
src_img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/mog_face_detection.jpg'
raw_result = tinymog_face_detection_func(src_img_path)
print('face detection output: {}.'.format(raw_result))

# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import draw_face_detection_no_lm_result
from modelscope.preprocessors.image import LoadImage
import cv2
import numpy as np

# load image from url as rgb order
src_img = LoadImage.convert_to_ndarray(src_img_path)
# save src image as bgr order to local
src_img  = cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR)
cv2.imwrite('src_img.jpg', src_img) 
# draw dst image from local src image as bgr order
dst_img = draw_face_detection_no_lm_result('src_img.jpg', raw_result)
# save dst image as bgr order to local
cv2.imwrite('dst_img.jpg', dst_img)
# show dst image by rgb order
import matplotlib.pyplot as plt
dst_img  = cv2.cvtColor(np.asarray(dst_img), cv2.COLOR_BGR2RGB)
plt.imshow(dst_img)
```
### 使用方式
- 推理：输入图片，如存在人脸则返回人脸位置，可检测多张人脸


### 目标场景
- 人脸相关的基础能力，可应用于人像美颜/互动娱乐/人脸比对等场景, 尤其适应于端上人脸应用场景


### 模型局限性及可能偏差
- 模型size较小，模型鲁棒性可能有所欠缺。
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 预处理
测试时主要的预处理如下：
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

### 模型训练流程
- 本模型使用8卡v100，使用SGD优化器，lr=0.02，在440/544epoch时降低10倍学习率，并在640epoch时产出模型，详见mmcv_tinymog.py


### 测试集
- WIDERFACE： 测试集已上传至ModelScope的DatasetHub，详情请见[WIDER_FACE](https://modelscope.cn/datasets/shaoxuan/WIDER_FACE)。

## 数据评估及结果
模型在WiderFace的验证集上客观指标如下：
| Method | Easy | Medium | Hard |
| ------------ | ------------ | ------------ | ------------ |
| TinyMog | - | - | - |

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

