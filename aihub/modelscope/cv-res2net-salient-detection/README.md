
# res2net-salient-detection 模型介绍
给定一张输入图像，通过目标边界信息引导找出图中的显著性目标，并输出视觉显著注意力图。

<img src=resources/test_salient_2.jpg width=10% /><img src=resources/test_salient_2_rst.jpg width=10% />

## 期望模型使用方式与适用范围
本模型适用范围较广，预测像素视觉显著注意程度，但不涉及图像中的语义信息。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
salient_detect = pipeline(Tasks.semantic_segmentation, model='damo/cv_res2net_salient-detection')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_salient_detection.jpg'
result = salient_detect(img_path)
import cv2
cv2.imwrite('./result.jpg',result[OutputKeys.MASKS])
```
### 模型局限性以及可能的偏差
- 算法模型预测像素视觉显著性程度，不涉及明确的语义表征。具体举例：一张图包含多个同语义区域，但是区域注意力值是不同的。
- 选取res2net50(Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License)作为特征提取网络，不支持商用。
## 数据集介绍
- DUTS
  该数据集包含10553张训练图像(DUTS-TR)和5019张测试图像(DUTS-TE)。所有训练图像均来自ImageNetDET训练/验证集，测试图像来自ImageNetDET测试集和SUN数据集，训练集和测试集都包含非常具有挑战性的显着性检测场景，真值由50个标注人员手动注释。详情见：http://saliencydetection.net/duts/

### 预处理
- 给定一张输入图像，分辨率归一化，颜色值减均值除方差归一化处理。
## 数据评估及结果
| DataSet  |     MAE     |    Sm   |  maxF~β  |   F^w~β    |
|:--------:|:-----------:|:-------:|:--------:| ---------- |
| DUTS-TE  |    0.034    |  0.897  |   0.887  |   0.853    |
| DUT-OMRON|    0.064    |  0.833  |   0.780  |   0.750    |
| SOD      |    0.093    |  0.797  |   0.840  |   0.766    |  
| PASCAL-S |    0.081    |  0.820  |   0.817  |   0.777    |
| ECSSD    |    0.029    |  0.932  |   0.946  |   0.927    |
| HKU-IS   |    0.027    |  0.918  |   0.931  |   0.907    |
| MSRA10K  |    0.056    |  0.893  |   0.899  |   0.871    |
## 引用
```BibTeX
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2021},
  doi={10.1109/TPAMI.2019.2938758},
}
```
