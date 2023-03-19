
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# <OSTrack>单目标跟踪算法模型介绍
对于一个输入视频，只需在第一帧图像中用矩形框指定待跟踪目标，单目跟踪算法将在整个视频帧中持续跟踪该目标，输出跟踪目标在所有图像帧中的矩形框信息。




## 模型描述
<img src="resources/intro.png" width="800" >

本模型是基于OSTrack方案的单目标跟踪框架，使用ViT作为主干网络进行训练，是One-Stream单目标跟踪算法。
本算法将特征抽取和匹配过程通过注意力机制同步进行，获得较高的目标跟踪精度。

## 期望模型使用方式以及适用范围

该模型适用于视频单目标跟踪场景，目前在LaSOT, TrackingNet, GOT-10K开源数据集上达到SOTA水平。

### 在线体验步骤

1. 点击“上传视频”按钮上传本地视频，或者直接使用默认示例视频。
2. 在视频图片中用鼠标框定目标。
3. 点击“执行测试”按钮，算法运行结束后，可在“测试结果”栏中播放查看跟踪结果视频。

### 如何使用模型

- 根据输入待跟踪视频和第一帧图像对应的待跟踪矩形框（x1, y1, x2, y2），可按照代码范例进行模型推理和可视化。

#### 代码范例
```python
from modelscope.utils.cv.image_utils import show_video_tracking_result
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model='damo/cv_vitb_video-single-object-tracking_ostrack')
video_path = "https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/dog.avi"
init_bbox = [414, 343, 514, 449] # the initial object bounding box in the first frame [x1, y1, x2, y2]
result = video_single_object_tracking((video_path, init_bbox))
show_video_tracking_result(video_path, result[OutputKeys.BOXES], "./tracking_result.avi")
print("result is : ", result[OutputKeys.BOXES])
```

### 模型局限性以及可能的偏差
- 在遮挡严重场景和背景中存在与目标高度相似的物体场景下，目标跟踪精度可能欠佳。
- 建议在有GPU的机器上进行测试，由于硬件精度影响，CPU上的结果会和GPU上的结果略有差异。


## 数据评估及结果
模型在GOT-10K的验证集上客观指标如下：

| Method | AO (Average Overlap) | SR (Success Rate) | 
| ------------ | ------------ | ------------ | 
| OSTrack | 0.872 | 0.963 | 

与一些经典方案在GOT-10K测试集的精度-效率分布图如下：

<img src="resources/performance.png" width="400" >


### 相关论文以及引用信息
本模型主要参考论文如下：

```BibTeX
@article{ye2022joint,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang},
  journal={arXiv preprint arXiv:2203.11991},
  year={2022}
}
```
