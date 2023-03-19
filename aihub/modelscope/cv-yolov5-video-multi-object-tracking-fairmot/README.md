
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# <FairMOT>多目标跟踪算法模型介绍
多目标跟踪算法通常由目标检测和目标重识别两个模块构成，FairMOT算法在单个网络中同时完成目标检测和重识别模块，可满足实时性要求。
<img src="resources/pipeline.png" width="800" >



## 期望模型使用方式以及适用范围

该模型适用于视频多目标跟踪行人场景，目前在2DMOT15数据集达到SOTA，在MOT16, MOT17, MOT20数据集上达到不错的效果。


#### 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.cv.video_multi_object_tracking.utils.visualization import show_multi_object_tracking_result

video_multi_object_tracking = pipeline(Tasks.video_multi_object_tracking, model='damo/cv_yolov5_video-multi-object-tracking_fairmot')
video_path = 'http://dmshared.oss-cn-hangzhou.aliyuncs.com/ljp/maas/mot_demo_resource/MOT17-03-partial.mp4?OSSAccessKeyId=LTAI5tC7NViXtQKpxFUpxd3a&Expires=2032715547&Signature=ROPQRkeOJqE3j8cBC0PEtkgdlzs%3D'
result = video_multi_object_tracking(video_path)
print('result is : ', result[OutputKeys.BOXES])
# show_multi_object_tracking_result(video_path, result[OutputKeys.BOXES], "mot_res.avi")
```

### 模型局限性以及可能的偏差
- 在遮挡严重场景和背景中存在与目标高度相似的物体场景下，目标跟踪精度可能欠佳。
- 建议在有GPU的机器上进行测试，由于硬件精度影响，CPU上的结果会和GPU上的结果略有差异。


## 训练数据介绍
本模型是基于以下开源数据集训练得到：
- [CrowdHuman](https://www.crowdhuman.org/)
- [MIX](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [MOT17](https://motchallenge.net/data/MOT17/)


## 数据评估及结果
模型在MOT17的测试集集上客观指标如下：
| Method    |  MOTA |
|--------------|-----------|
|FairMOT  | 68.5 |



### 相关论文以及引用信息
本模型主要参考论文如下：

```BibTeX
@article{zhang2021fairmot,
  title={Fairmot: On the fairness of detection and re-identification in multiple object tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={International Journal of Computer Vision},
  volume={129},
  pages={3069--3087},
  year={2021},
  publisher={Springer}
}
```