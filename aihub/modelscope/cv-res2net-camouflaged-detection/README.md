
# res2net-camouflaged-detection 模型介绍
给定一张输入图像，找出图中的伪装色目标，并输出视觉显著注意力图。

<img src=resources/test_camouflag_0.jpg width=20% /><img src=resources/test_camouflag_0_rst.jpg width=20% />

<img src=resources/test_camouflag_3.jpg height=25% /><img src=resources/test_camouflag_3_rst.jpg height=25% />

## 期望模型使用方式与适用范围
本模型是针对伪装色目标（颜色、纹理等统计特征与所处环境一致）进行定位，并输出视觉区域图，模型是在学术数据集上训练，需要待检测图像目标类别在数据集范围之内。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
camouflag_detect = pipeline(Tasks.semantic_segmentation, model='damo/cv_res2net_camouflaged-detection')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_camouflag_detection.jpg'
result = camouflag_detect(img_path)
import cv2
cv2.imwrite('./result.jpg',result[OutputKeys.MASKS])
```
### 模型局限性以及可能的偏差
- 模型是在学术数据集上训练，需要待检测图像目标类别在学术数据集类别内。
- 选取res2net50(Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License)作为特征提取网络，不支持商用。
## 数据集介绍
- COD10K
  数据集包含10000幅图像，覆盖各种自然场景中的伪装对象，超过78个对象类别，下载地址见：https://mmcheng.net/code-data/
- CAMO
  数据集包含1250图像(1000张训练，250测试)，覆盖八个类别，主要是伪装的动物与人数据。
- CHAMELEON
  包含75张测试数据。
- NC4K
  互联网下载的4121张测试图像，可以用来评估现有模型的泛化能力。

### 预处理
- 给定一张输入图像，分辨率归一化，颜色值减均值除方差归一化处理。
## 数据评估及结果
| DataSet  |     MAE     |    Sm   |  maxF~β  |   F^w~β    |
|:--------:|:-----------:|:-------:|:--------:| ---------- |
| CAMO     |    0.068    |  0.838  |   0.823  |   0.780    |
| CHAMELEON|    0.026    |  0.907  |   0.888  |   0.858    |
| COD10K   |    0.031    |  0.843  |   0.789  |   0.739    |  
| NC4K     |    0.042    |  0.862  |   0.841  |   0.797    |
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
