# 17点人体关键点检测模型

输入一张人物图像，实现端到端的人体关键点检测，输出视频中每一帧图像人体的17点人体3D关键点坐标。

## 17点人体关键点
![3D人体关键点定义](assets/example1.png)

## 3D人体关键点系列模型

| [<img src="assets/HDFormer.jpg" width="300px">](https://modelscope.cn/models/damo/cv_hdformer_body-3d-keypoints_video/summary) |	[<img src="assets/CanonicalPose3D.jpg" width="300px">](https://modelscope.cn/models/damo/cv_canonical_body-3d-keypoints_video/summary)  |
 |:--:|:--:|
| [HDFormer](https://modelscope.cn/models/damo/cv_hdformer_body-3d-keypoints_video/summary) |	 [CannoicalPose3D](https://modelscope.cn/models/damo/cv_canonical_body-3d-keypoints_video/summary) 	 |

## 模型描述
该任务是单目相机下的3D人体关键点检测框架，通过端对端的快速推理，可以得到视频中的人体3D关键点坐标。其中2D人体检测基于[此2D人体关键点检测模型.](https://modelscope.cn/#/models/damo/cv_hrnetv2w32_body-2d-keypoints_image/summary)
</br>
</br>
</br>
![](./assets/CanonicalPose3D.jpg)
</br>
</br>

本模型参考TPNet改进VideoPose3D，基于2D图像空间校准坐标，设计新网络优化3D全局轨迹。

## 使用范围和应用场景
使用范围:
- 包含人体的视频，人体分辨率大于100x100，总体图像分辨率小于1000x1000，视频大小不超过20MB，视频时长不超过30s。

应用场景:
1. 动作计数：可用于AI体测场景；
2. 动作匹配打分：可用于娱乐、健身场景中等场景，实现AI动作纠错与负反馈；
3. 人体动作识别：可用于监控、医疗健康等场景，通过3D人体姿态分析人体行为参数；
4. 虚拟驱动：基于3D人体姿态驱动3D虚拟形象，实现低成本UCG内容生成

### 如何使用

在ModelScope框架上，提供输入视频，即可通过简单的Pipeline调用来完成人体关键点检测任务。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_canonical_body-3d-keypoints_video'
body_3d_keypoints = pipeline(Tasks.body_3d_keypoints, model=model_id)
output = body_3d_keypoints('https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/Walking.54138969.mp4')
print(output)
```

##### 参数配置
- 调整测试视频的测试帧数：在`configuration.json`里面指定的测试视频中使用的帧数：`model.INPUT.MAX_FRAME`；
- 调整预测结果渲染3D视频的视角：在`configuration.json`里面指定`render`字段中的方位角和偏转角；
#### 输出范例
```json
{
  "keypoints": [		// 相机坐标系下的3D姿态关键点坐标
    	[[x, y, z]*17],	// 每行为一帧图片的预测结果
    	[[x, y, z]*17],
    	...
    ],
  "timestamps": [ // 每一帧测试视频对应的时间戳
    "00:00:0.23",
    "00:00:0.56",
    "00:00:0.69",
    ...
  ],
  "output_video": "xxx" // 渲染推理结果的视频二进制文件数据，可选，取决于模型配置文件中是否配置"render"字段。
}
```

### 模型局限性以及可能的偏差

- 输入图像存在人体严重残缺的情形下，模型会出现人体或点位误检和漏检的现象；
- 高度运动模糊的情形下，模型会出现人体或点位误检和漏检的现象；
- 由于3D人体关键点检测和相机的内外参数有一定关系，输入任意视频片段进行测试可能效果不佳；当前测试视频的相机内外参配置信息在`configuration.json`中的 `model.INPUT` 字段进行定义。



## 训练数据介绍
训练数据：[Human3.6M](http://vision.imar.ro/human3.6m)。
```
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu, Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = { IEEE Transactions on Pattern Analysis and Machine Intelligence}，
  publisher = {IEEE Computer Society}，
  year = {2014}
}

```

## 数据评估及结果

| MPJPE |	 P-MPJPE |	 N-MPJPE |	 MPJVE |
 |---|---|---|---|
| 40.400 |	 29.400 	 |37.700 |	 1.900 |

### 模型效果
输出的3D关键点可视化结果如下：

<div align="center">
  <img src="assets/frame1.jpg" width="500" />
  <center>输入视频截图</center>
  </br>
  <img src="assets/result.png" width="500" />
  <center>预测结果</center>
</div>
