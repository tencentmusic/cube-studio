
# 全景图深度估计算法介绍

## 任务
输入一张单目全景RGB图像，全景深度估计算法将分析场景三维结构、输出图像对应的稠密深度图.

## 模型描述

本模型基于**UniFuse: Unidirectional Fusion for 360° Panorama Depth Estimation**，是该算法的官方模型。

技术细节请见：

[arXiv](https://arxiv.org/abs/2102.03550), [Demo](https://youtu.be/9vm9OMksvrc)


## 如何使用

### 代码示例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import depth_to_color


task = 'panorama-depth-estimation'
model_id = 'damo/cv_unifuse_panorama-depth-estimation'

input_location = 'data/test/images/panorama_depth_estimation.jpg'
estimator = pipeline(Tasks.panorama_depth_estimation, model=model_id)
result = estimator(input_location)
depth_vis = result[OutputKeys.DEPTHS_COLOR]
cv2.imwrite('result.jpg', depth_vis)
```

### 适用范围

输入一张单目全景RGB图像，图像投影模型为Equirectangular，即长宽比为2:1，建议分辨率为1024x512.


## 模型精度
在Matterport3D上的结果为
| Model | MAE | Abs Rel | RMSE | RMSElog | δ<1.25 | δ<1.25<sup>2</sup> | δ<1.25<sup>3</sup> | 
| :--- | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: | 
|Unifuse | 0.2814 | 0.1063 | 0.4941 | 0.0701 | 88.97 | 96.23 | 98.3 |

## Bibtex

```
@article{jiang2021unifuse,
      title={UniFuse: Unidirectional Fusion for 360$^{\circ}$ Panorama Depth Estimation}, 
      author={Hualie Jiang and Zhe Sheng and Siyu Zhu and Zilong Dong and Rui Huang},
	  journal={IEEE Robotics and Automation Letters},
	  year={2021},
	  publisher={IEEE}
}
```
