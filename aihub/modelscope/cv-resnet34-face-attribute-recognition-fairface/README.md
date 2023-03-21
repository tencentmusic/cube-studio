
# FairFace 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸属性识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizeFace&spm=a2cio.27993362)、[表情识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizeExpression&spm=a2cio.27993362)。

人脸属性模型FairFace


## 模型描述

FairFace是结构简单性能不错的人脸属性模型（[代码地址](https://github.com/dchen236/FairFace)), 被WACV2021录取([论文地址](https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf), 是人脸属性领域近几年高引文章之一。网络结构比较简单，backbone是标准的ResNet34结构，后面接一个fc层，对输入图片的人脸预测其年龄区间和性别。


## 模型效果
![模型效果](FairFace.png)

## 模型使用方式和使用范围
本模型可以检测输入图片中人的性别和年龄范围:[0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition, 'damo/cv_resnet34_face-attribute-recognition_fairface')
src_img_path = 'data/test/images/face_recognition_1.png'
raw_result = fair_face_attribute_func(src_img_path)
print('face attribute output: {}.'.format(raw_result))

# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import draw_face_attribute_result
from modelscope.preprocessors.image import LoadImage
import cv2
import numpy as np

# load image from url as rgb order
src_img = LoadImage.convert_to_ndarray(src_img_path)
# save src image as bgr order to local
src_img  = cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR)
cv2.imwrite('src_img.jpg', src_img) 
# draw dst image from local src image as bgr order
dst_img = draw_face_attribute_result('src_img.jpg', raw_result)
# save dst image as bgr order to local
cv2.imwrite('dst_img.jpg', dst_img)
# show dst image by rgb order
import matplotlib.pyplot as plt
dst_img  = cv2.cvtColor(np.asarray(dst_img), cv2.COLOR_BGR2RGB)
plt.imshow(dst_img)
```
### 使用方式
- 推理：输入图片，如存在人脸则返回人的性别以及年龄区间。


### 目标场景
- 人脸相关的基础能力，可应用于视频监控/人像美颜/互动娱乐等场景


### 模型局限性及可能偏差
- 训练数据仅包括FairFace数据集，模型鲁棒性可能有所欠缺。
- 目前只支持单人的属性识别。
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 预处理
测试时主要的预处理如下：
- Resize：图像resize到224x224
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
本模型及代码来自开源社区([地址](https://github.com/dchen236/FairFace))，请遵守相关许可。

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@article{karkkainen2019fairface,
      title={Fairface: Face attribute dataset for balanced race, gender, and age},
        author={K{\"a}rkk{\"a}inen, Kimmo and Joo, Jungseock},
          journal={arXiv preprint arXiv:1908.04913},
            year={2019}
            }
```
