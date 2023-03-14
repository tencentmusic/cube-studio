

# 视频目标检测 
<div align=center>
<img src="res/latency-aware.gif">
</div>
自动驾驶实时视频检测模型, 把周围环境检测问题，转化为将来环境预测问题，从问题定义的层面解决自动驾驶中环境感知时延的问题。该任务定义为流感知（Streaming Perception）问题.

Perceive the world by predicting!

## 模型描述

基于StreamYOLO的实时通用检测模型，支持8类交通目标检测。StreamYOLO基于YOLOX模型，使用Dual-Flow Perception特征融合模块，learns 特征层面的时序关系，提高环境感知预测的能力。与此同时，StreamYOLO设计了一个Trend-Aware Loss 去感知物体运动变化强度，用以加权物体预测的回归，使运动剧烈变化物体获得更高的回归权重，从而获得更好的预测结果。

<p align='center'>
  <img src='res/train.png' width='721'/>
</p>


## 模型使用方式以及适用范围

- 自动驾驶场景交通目标预测/检测
- 自动驾驶场景决策支持前置感知算法
- 作为自动驾驶场景pretrained model初始化模型

### 如何使用

开始你的模型探索之旅！

Play the model with a few line codes !

#### 代码范例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

model_id = 'damo/cv_cspnet_video-object-detection_streamyolo'
test_video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/test_realtime_vod.mp4'
# 初始化实时检测pipeline
realtime_video_object_detection = pipeline(
    Tasks.video_object_detection, model=model_id)

# 进行实时检测 
result = realtime_video_object_detection(test_video)
if result:
    bboxes_list = result[OutputKeys.BOXES]
    print(bboxes_list)
else:
    raise ValueError('process error')
```

## 模型局限性以及可能的偏差
- 对于非自动驾驶前置摄象机场景会出现明显检测性能下降的情况。
- 目前模型仅限于pipeline调用，尚未支持Finetune和Evaluation。
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试。

## 训练数据介绍

Argoverse-HD 数据集是最近提出的自动驾驶数据集，该数据集与其他的自动驾驶相比，数据规模中等，复杂程度较高，是一个较有代表性的数据集。更重要的是，Argoverse-HD 是第一个提出流感知任务的数据集，并且设计了Stream AP评测标准。该标准将感知时延充分考虑，实现对模型的性能-速度的全面、有效评价。
<div align=center>
<img src="res/datasets.png">
</div>

## 训练
本模型暂时不支持finetune， 具体离线训练细节如下：
- 在Argoverse-HD 上训练十五个epoch
- 使用SGD优化算法，线性 LR 策略
- 使用flip数据增强， 多尺度训练增强

## 输入预处理
- 输入图像根据短边resize到640后，padding 为640x960的矩形进行推理
- 图像归一化


## 数据评估及结果

|Model |size |velocity | sAP<br>0.5:0.95 | sAP50 |sAP75| weights | COCO pretrained weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: | :----: |
|[StreamYOLO-l](https://arxiv.org/pdf/2207.10433.pdf)    |600×960  |1x  |36.9 |58.1| 37.5 |[official](https://github.com/yancie-yjr/StreamYOLO/releases/download/0.1.0rc/l_s50_one_x.pth) |[official](https://github.com/yancie-yjr/StreamYOLO/releases/download/0.1.0rc/yolox_l.pth) |


## 相关论文以及引用信息
```
@inproceedings{streamyolo,
  title={Real-time Object Detection for Streaming Perception},
  author={Yang, Jinrong and Liu, Songtao and Li, Zeming and Li, Xiaoping and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5385--5395},
  year={2022}
}

@article{yang2022streamyolo,
  title={StreamYOLO: Real-time Object Detection for Streaming Perception},
  author={Yang, Jinrong and Liu, Songtao and Li, Zeming and Li, Xiaoping and Sun, Jian},
  journal={arXiv preprint arXiv:2207.10433},
  year={2022}
}

```

