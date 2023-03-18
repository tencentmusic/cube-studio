
# u2net-salient-detection模型介绍
给定一张输入图像，输出视觉显著注意力图。

<img src=resources/test_salient_3.jpg width=25% /><img src=resources/test_salient_3_salient.jpg width=25% />


## 期望模型使用方式与适用范围
本模型适用范围较广，预测像素视觉显著注意程度，但不涉及图像中的语义信息。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
salient_detect = pipeline(Tasks.semantic_segmentation, model='damo/cv_u2net_salient-detection')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_salient_detection.jpg'
result = salient_detect(img_path)
import cv2
cv2.imwrite('./result.jpg',result[OutputKeys.MASKS])
```
### 模型局限性以及可能的偏差
- 算法模型预测像素视觉显著性程度，不涉及明确的语义表征。具体举例：一张图包含多个同语义区域，但是区域注意力值是不同的。
## 训练数据介绍
- DUTS
  该数据集包含10553张训练图像(DUTS-TR)和5019张测试图像(DUTS-TE)。所有训练图像均来自ImageNetDET训练/验证集，测试图像来自ImageNetDET测试集和SUN数据集，训练集和测试集都包含非常具有挑战性的显着性检测场景，真值由50个标注人员手动注释。详情见：http://saliencydetection.net/duts/
### 预处理
- 给定一张输入图像，分辨率归一化，颜色值减均值除方差归一化处理。
## 数据评估及结果
| DataSet  |     MAE     |    Sm   |  maxF~β  |   F^w~β    |
|:--------:|:-----------:|:-------:|:--------:| ---------- |
| DUTS-TE  |    0.044    |  0.861  |   0.873  |   0.804    |
| DUT-OMRON|    0.054    |  0.847  |   0.823  |   0.757    |
| SOD      |    0.108    |  0.786  |   0.861  |   0.748    |  
| ECSSD    |    0.033    |  0.928  |   0.951  |   0.910    |
| HKU-IS   |    0.031    |  0.916  |   0.935  |   0.890    |
## 引用
```BibTeX
@article{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```
