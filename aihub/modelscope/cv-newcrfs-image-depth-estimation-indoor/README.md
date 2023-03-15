
# 单目深度估计算法介绍

## 任务
输入一张单目RGB图像，单目深度估计算法将分析场景三维结构、输出图像对应的稠密深度图

## 模型描述

本模型基于**NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation**算法，是该算法的官方模型。

技术细节请见：

**NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation** <br />
Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, Ping Tan <br />
**CVPR 2022** <br />
**[[Project Page](https://weihaosky.github.io/newcrfs/)]** | 
**[[Paper](https://arxiv.org/abs/2203.01502)]** |
**[[中文解读](https://zhuanlan.zhihu.com/p/522214395)]**  <br />


<p float="left">
  &emsp;&emsp; <img src="description/intro.png" width="400" />
</p>

![Output1](description/output_nyu2_compressed.gif)


## 如何使用

### 代码示例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import depth_to_color


task = 'image-depth-estimation'
model_id = 'damo/cv_newcrfs_image-depth-estimation_indoor'

input_location = 'data/test/images/image_depth_estimation.jpg'
estimator = pipeline(Tasks.image_depth_estimation, model=model_id)
result = estimator(input_location)
depth_vis = result[OutputKeys.DEPTHS_COLOR]
cv2.imwrite('result.jpg', depth_vis)
```

### 适用范围

默认输入图片的摄像机参数应与训练数据集（NYUv2）保持一直, 即分辨率为640x480，内参为
```
518.8579,      0.0,   320
     0.0, 518.8579,   240
     0.0,      0.0,   0.0
```
如输入图像不一致，请将输入图片矫正为上述参数，否则会影响结果准确性


## 模型精度
在NYUv2上的结果为
| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | a1 | a2 | a3| SILog| 
| :--- | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|NYUv2 | 0.0952 | 0.0443 | 0.3310 | 0.1185 | 0.923 | 0.992 | 0.998 | 9.1023 |


## Demo Video

![Output1](description/output_nyu1_compressed.gif)


## Bibtex

```
@inproceedings{yuan2022newcrfs,
  title={NeWCRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation},
  author={Yuan, Weihao and Gu, Xiaodong and Dai, Zuozhuo and Zhu, Siyu and Tan, Ping},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2022}
}
```