
# 通用实时检测模型

实时目标检测-自动驾驶领域模型是专注于自动驾驶场景的目标检测模型。


## 模型描述

YOLOX-PAI是阿里云机器学习平台PAI的开源计算机视觉代码库[EasyCV](https://github.com/alibaba/EasyCV)中集成的YOLOX算法, 提供快速、精确的目标检测能力。本模型为YOLOX-PAI的小规模模型，使用了RepVGG结构作为backbone，ASFF作为neck，TOOD作为head，基于公开数据集Waymo、BDD100K、Nuimage100K训练，目前支持车辆目标检测。

下图为检测效果样例图：

<img src="res/demo_res.jpg" width="600">

## 期望模型使用方式以及适用范围

- 自动驾驶场景目标定位与识别。
- 自动驾驶场景车辆粗略计数。
- 作为其他自动驾驶场景算法的前置算法，如跟踪算法等。

### 如何使用


在ModelScope框架上，可以通过ModelScope的pipeline进行调用.

#### 代码范例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
```python
# numpy >= 1.20
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

realtime_detector = pipeline(Tasks.image_object_detection, model='damo/cv_yolox_image-object-detection-auto')
result = realtime_detector('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/auto_demo.jpg')
# print predicted results, including scores, labels, boxes
print(result)
```

### 模型局限性以及可能的偏差
  - 对于尺寸特别小的车辆和行人可能会存在检出效果不佳的情况，建议对过小检出目标的进行限制。
  - 目前模型仅限于pipeline调用，尚未支持Finetune和Evaluation。
  - 非自动驾驶场景性能会产生显著下降，如监控场景、无人机视角场景等。
  - 当前版本在python 3.7环境测试通过，其他环境下可用性待测试。

## 训练数据介绍

<figure>
<img src="res/waymo.png" width="800">  
</figure>


<figure>
<img src="res/nuimages.png" width="800">  
</figure>


<figure>
<img src="res/BDD.png" width="800">  
</figure>


本模型基于Waymo、Nuimage100K、BDD100K数据集中所有包含车辆的目标检测训练集图片及标注进行训练。

## 模型训练流程

模型在线训练暂不支持。部分关键训练细节如下：
- 使用 SGD 优化算法，cos LR scheduler，warmup策略。
- 训练迭代为 300 epoch，其中最后15个epoch关闭数据增强。
- Mosaic，颜色增强等策略被应用到训练预处理中。

## 输入预处理

- 输入图像根据长边resize到640后，padding 为640x640的矩形进行推理
- 图像归一化

## 数据评估及结果
|Model |size |mAP<sup>val<br>0.5:0.95 | Speed V100<br>(ms) fp16 bs32 | Params<br>(M) |FLOPs<br>(G)|
| ------        |:---:  | :---:       |:---:     |:---:  | :---: |
|[YOLOX-PAI](https://arxiv.org/pdf/2208.13040.pdf)    |640  |43.9 |1.15      |23.7 | 49.9 |


## 引用

如您的相关著作、作品使用了该模型，请引用以下信息：

```
@article{zou2022yolox,
  title={YOLOX-PAI: An Improved YOLOX Version by PAI},
  author={Zou, Xinyi and Wu, Ziheng and Zhou, Wenmeng and Huang, Jun},
  journal={arXiv preprint arXiv:2208.13040},
  year={2022}
}
```