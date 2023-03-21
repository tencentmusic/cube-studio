# 17点人体关键点检测模型

输入一段包含人物的视频，实现端到端的人体关键点检测，输出视频中每一帧图像人体的17点人体3D关键点坐标。

## 3D人体关键点系列模型

| [<img src="https://modelscope.cn/api/v1/models/damo/cv_hdformer_body-3d-keypoints_video/repo?Revision=master&FilePath=assets/arch.jpg&View=true" width="300px">](https://modelscope.cn/models/damo/cv_hdformer_body-3d-keypoints_video/summary) |	[<img src="assets/CanonicalPose3D.jpg" width="300px">](https://modelscope.cn/models/damo/cv_canonical_body-3d-keypoints_video/summary)  |
 |:--:|:--:|
| [HDFormer](https://modelscope.cn/models/damo/cv_hdformer_body-3d-keypoints_video/summary) |	 [CannoicalPose3D](https://modelscope.cn/models/damo/cv_canonical_body-3d-keypoints_video/summary) 	 |

## 模型描述
该任务是单目相机下的3D人体关键点检测框架，通过端对端的快速推理，可以得到视频中的人体3D关键点坐标。其中2D人体检测基于[2D人体关键点检测模型.](https://modelscope.cn/#/models/damo/cv_hrnetv2w32_body-2d-keypoints_image/summary)

</br>
<div align="center">
  <img src="https://modelscope.cn/api/v1/models/damo/cv_hdformer_body-3d-keypoints_video/repo?Revision=master&FilePath=assets/arch.jpg&View=true" width="800" />
  </br>
</div>
</br>

本模型HDFormer是一种U-Shape结构的3D人体姿态估计模型，包含3个不同的stage：下采样阶段、上采样阶段和合并阶段。本模型结合了joint<->joint, bone<->joint 和 hyperbone<->joint的特征交互。HDFormer相比于Transformer结构的3D人体姿态估计模型，具有更高的预测精度和推理效率。

</br>
<div align="center">
  <img src="https://modelscope.cn/api/v1/models/damo/cv_hdformer_body-3d-keypoints_video/repo?Revision=master&FilePath=assets/block.jpg&View=true" width="800" />
  </br>
</div>
</br>

HDFormer结构主要包含High-order Directed Tranformer block。该结构中主要包含First-order Attention block, Hyperbone Representation和Cross-attention三种结构。其中First-order Attention block用来提取所有关节点之间的空间结构信息；Hyperbone Representation用来描述不同阶数的人体结构特征；Cross-attention用来提取关节点与高阶骨骼特征之间的关系。

## 使用范围和应用场景
使用范围:
- 包含人体的视频，人体分辨率大于100x100，总体图像分辨率小于1000x1000，视频大小不超过20MB，视频时长不超过30s，视频帧数量超过100帧。

应用场景:
1. 动作计数：可用于AI体测场景；
2. 动作匹配打分：可用于娱乐、健身场景中等场景，实现AI动作纠错与负反馈；
3. 人体动作识别：可用于监控、医疗健康等场景，通过3D人体姿态分析人体行为参数；
4. 虚拟驱动：基于3D人体姿态驱动3D虚拟形象，实现低成本UCG内容生成

### 如何使用

在ModelScope框架上，提供输入视频，即可通过简单的Pipeline调用来完成人体关键点检测任务。暂不支持CPU。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_hdformer_body-3d-keypoints_video'
body_3d_keypoints = pipeline(Tasks.body_3d_keypoints, model=model_id)
output = body_3d_keypoints('https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/Walking.54138969.mp4')
print(output)
```

##### 参数配置
- 调整测试视频的测试帧数：在`configuration.json`里面指定的测试视频中使用的帧数：`model.INPUT.max_frame`；
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
- 模型训练时，时间维度上的视频帧数为96，不足96帧的情况下没有进行插帧处理，因此测试视频帧数量需要大于等于96帧。
  



## 训练数据介绍
训练数据：[Human3.6M](http://vision.imar.ro/human3.6m)。


## 数据评估及结果
### Human3.6M

| MPJPE(mm) |	 2d_gt(T=96) |	 cpn(T=96) |	 hrnet(T=96) |
 |---|---|---|---|
| HDFormer |	 21.6 	 |42.6 |	 40.3 |
### MPI-INF-3DHP

|  |	 PCK[↑]| AUC[↑]| MPJPE[↓]|
|---|---|---|---|
| HDFormer |	 98.7% | 72.9% | 37.2mm |

### 推理效率

| Method |	 MPJPE[↓]| Params| Latency | Frames |
 |---|---|---|---| ---|
| U-CondDGCN[^u_conddgcn] |	 22.7 	|3.4M | 0.6ms |	 96 |
| MixSTE[^mixste] |	 25.9 	|33.7M | 2.6ms |	 81 |
| MixSTE |	 21.6 	|33.8M | 8.0ms |	 243 |
| HDFormer |	 21.6 	|3.7M | 1.3ms |	 96 |

*测试环境为V100 16GB GPU。*

> [^mixste]: Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. Mixste: Seq2seq mixed spatio-temporal encoder for 3d human pose estimation in video. In IEEE Conference on Computer Vision and Pat- tern Recognition, (CVPR), pages 13222–13232, 2022.
> [^u_conddgcn]: Wenbo Hu, Changgong Zhang, Fangneng Zhan, Lei Zhang, and Tien-Tsin Wong. U-CondDGCN: Conditional Directed Graph Convolution for 3D Human Pose Estimation. arXiv, 2021

### 模型效果
输出的3D关键点可视化结果如下：

<div align="center">
  <img src="https://modelscope.cn/api/v1/models/damo/cv_hdformer_body-3d-keypoints_video/repo?Revision=master&FilePath=assets/frame1.jpg&View=true" width="500" />
  <center>输入视频截图</center>
  </br>
  <img src="assets/result.png" width="500" />
  <center>预测结果</center>
</div>

# 相关论文以及引用信息

```
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu, Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = { IEEE Transactions on Pattern Analysis and Machine Intelligence}，
  publisher = {IEEE Computer Society}，
  year = {2014}
}

@article{chen2023-hdformer,
  title = {HDFormer: High-order Directed Transformer for 3D Human Pose Estimation},
  author = {Chen, Hanyuan and He, Jun-Yan and Xiang, Wangmeng and Liu, Wei and Cheng, Zhi-Qi and Liu, Hanbing and Luo, Bin and Geng, Yifeng and Xie, Xuansong},
  year = {2023},
  eprint = {2302.01825},
  doi = {10.48550/arXiv.2302.01825},
}
```
