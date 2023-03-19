
# YOLOPV2全景驾驶感知模型介绍
YOLOPv2 适用于自动驾驶场景下的实时全景驾驶感知, 同时执行三种不同的任务，分别为车辆检测，可行驶区域分割以及车道线分割。（注：在线体验功能目前存在两个已知问题：1.仅支持1280x720图像上传。2.未显示车辆检测情况，这两个问题将在315左右解决）

## 模型描述
YOLOPv2([官方代码](https://github.com/CAIC-AD/YOLOPv2)) 是[YOLOP](https://arxiv.org/abs/2108.11250)(You Only Look Once for Panoptic Driving Perception)的升级版，输入一张图像，输出所有车辆区域检测框、可行驶区域(Drivable area)和车道线(Lane Line)分割图。可视化结果如下。

<img src="./assets/image_driving_perception.jpg" width = "576" height = "324" alt="source"/> <img src="./assets/result.jpg" width = "576" height = "324" alt="result"/>

模型整体架构图如下:

<img src="./assets/yolopv2_arch.png"  alt="YOLOPv2模型架构"/>

## 使用方式和范围

使用方式：
- 输入任意分辨率图像，返回图像中的车辆坐标，可行驶区域及车道线二值化（0，1）数组，支持CPU/GPU 推理。

使用范围:
- 本模型适用于车载镜头视角下的自动驾驶场景，适合2D image 输入的任务。

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_driving_perception.jpg'
image_driving_perception_pipeline = pipeline(Tasks.image_driving_perception,
                                        model='damo/cv_yolopv2_image-driving-perception_bdd100k')
result = image_driving_perception_pipeline(img_path)
# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import show_image_driving_perception_result
import matplotlib.pyplot as plt
import cv2
from modelscope.preprocessors.image import LoadImage
output = './result.jpg'
img = LoadImage.convert_to_ndarray(img_path)
img_res = show_image_driving_perception_result(img, result,out_file=output)
plt.figure()
plt.imshow(img_res)
```

### 模型局限性以及可能的偏差

- 部分非常规图片（鱼眼环视图)或车辆遮挡严重可能会影响输出结果
- 模型暂不支持行人检测

## 训练数据介绍

- 使用BDD100K 数据集

## 数据评估及结果

模型在BDD100K验证集上分别对车辆检测、可驾驶区域分割及车道线分割进行评估，结果如下:

车辆检测
|        Model       |     mAP@0.5 (%)   |Recall (%)  |
|:------------------:|:------------:|:---------------:|
|     `MultiNet`     |        60.2      |   81.3     |  
|      `DLT-Net`     |        68.4      |  89.4     |
|   `Faster R-CNN`   |          55.6      | 77.2     |
|      `YOLOv5s`     |         77.2      | 86.8     |
|       `YOLOP`      |        76.5      | 89.2     |   
|     `HybridNets`   |          77.3      | **92.8**   | 
|    **`YOLOPv2`**   |       **83.4**(+6.1)  |   91.1(-1.7) |

可驾驶区域分割
|       Model      | Drivable mIoU (%) |
|:----------------:|:-----------------:|
|    `MultiNet`    |        71.6       |   
|     `DLT-Net`    |        71.3       | 
|     `PSPNet`     |        89.6       | 
|      `YOLOP`     |        91.5       | 
|     `HybridNets` |        90.5       | 
|     `YOLOPv2`    |   **93.2(+1.7)**  |

车道线分割
|      Model       | Accuracy (%) | Lane Line IoU (%) |
|:----------------:|:------------:|:-----------------:|
|      `Enet`      |     34.12    |       14.64       |
|      `SCNN`      |     35.79    |       15.84       |
|    `Enet-SAD`    |     36.56    |       16.02       |
|      `YOLOP`     |     70.5     |        26.2       |
|   `HybridNets`   |     85.4     |        **31.6**   |
|    **`YOLOPv2`** |   **87.3(+1.9)**|      27.2(-4.4)|


## 引用

```BibTeX
@article{han2022yolopv2,
  title={YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception},
  author={Cheng Han, Qichao Zhao, Shuyi Zhang, Yinzi Chen, Zhenlin Zhang, Jinwei Yuan},
  journal={arXiv preprint arXiv:2208.11434},
  year={2022}
}
```

### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_yolopv2_image-driving-perception_bdd100k.git
```