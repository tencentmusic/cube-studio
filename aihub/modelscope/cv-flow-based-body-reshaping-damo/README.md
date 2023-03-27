
# 基于光流的人体美型(FBBR)

### [论文](https://arxiv.org/abs/2203.04670) ｜ [github](https://github.com/JianqiangRen/FlowBasedBodyReshaping)

给定一张单个人物图像（半身或全身），无需任何额外输入，端到端地实现对人物身体区域（肩部，腰部，腿部等）的自动化美型处理。相关论文发表在CVPR2022上。

其生成效果如下所示：

![生成效果](description/demo.gif)


## 模型描述

该任务借助于人体姿态识别模型提取的骨骼结构信息，对输入图像提取与结构耦合的深度特征，预测出光流形变场，并映射回原始图像，最终实现了业内首个端到端的自适应美型功能。

![模型结构](https://modelscope.cn/api/v1/models/damo/cv_flow-based-body-reshaping_damo/repo?Revision=master&FilePath=description/framework.jpg&View=true)

## 使用方式和范围

使用方式：
- 直接推理，在任意真实人物图像上进行直接推理。

使用范围:
- 包含单个人物身体的图片，支持半身或全身，支持正面、侧面、坐姿等多种姿态，图像分辨率大于100x100，小于3000×3000。

目标场景:
- 需要进行身体美型的场景，如摄影后期，广告宣传等。

### 如何使用

在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来使用。

#### 代码范例
```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

image_body_reshaping = pipeline(Tasks.image_body_reshaping, 
                       model='damo/cv_flow-based-body-reshaping_damo')
image_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_body_reshaping.jpg'
result = image_body_reshaping(image_path)

cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
print('finished!')

```

### 模型局限性以及可能的偏差

- 模型训练数据有限，部分非常规人体图像或者人体在图像中占比过小可能会影响美型效果。

- 少数复杂姿态和被遮挡的人体图像会无法准确检测出骨骼信息，从而影响到最终美型效果。

## 训练数据介绍

- [BR-5K数据集](https://github.com/JianqiangRen/FlowBasedBodyReshaping), 包含5000张高质量人像美型数据，类型涵盖不同种族，年龄，姿态和服装。

- 数据从[unsplash](https://unsplash.com)网站收集，并由专业设计师标注、筛选得到。

## 模型推理流程

### 预处理

- 人体骨骼点检测。
- 人体区域提取&对齐，得到256x256大小的图像用于模型推理。


## 数据评估及结果

使用BR-5K数据集进行评测，在SSIM/PSNR/LPIPS/用户偏好等指标上均达SOTA结果。

| Method | SSIM | PSNR | LPIPS | User Preference | 
| ------------ | ------------ | ------------ | ------------ | ------------ |
| Origin | 0.8339  | 24.4916 | 0.0823  | N.A. | 
| [FAL](https://arxiv.org/abs/1906.05856)    | 0.8261  | 24.1841 | 0.0837  | 14.4%| 
| [ATW](https://arxiv.org/abs/2008.00362)    | 0.8316  | 24.6332 | 0.0805  | 9.8% | 
| [pix2pixHD](https://arxiv.org/abs/1711.11585) | 0.7271  | 21.8381 | 0.2800  | 3.6% | 
| [GFLA](https://arxiv.org/abs/2003.00696)   | 0.6649  | 21.4796 | 0.6136  | 1.8% |
| Ours   | **0.8354** | **24.7924** | **0.0777** | **70.4%** |

## 引用
如果该模型对你有所帮助，请考虑引用如下相关论文：

```BibTeX
@inproceedings{ren2022structure,
  title={Structure-Aware Flow Generation for Human Body Reshaping},
  author={Ren, Jianqiang and Yao, Yuan and Lei, Biwen and Cui, Miaomiao and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR2022)},
  pages={7754--7763},
  year={2022}
}
```
