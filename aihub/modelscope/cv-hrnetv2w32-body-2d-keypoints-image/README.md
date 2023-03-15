# 15点人体关键点检测模型

输入一张人物图像，实现端到端的人体关键点检测，输出图像中所有人体的15点人体关键点坐标、点位置信度和人体检测框，点位顺序如下图所示。

## 15点人体关键点
![人体关键点定义](assets/keypoints.png)

## 模型描述
该任务采用自顶向下的人体关键点检测框架(如下图)，通过端对端的快速推理可以得到图像中的人体关键点。其中人体关键点模型基于HRNet的backbone，充分利用多分辨率的特征融合，良好支持日常人体姿态，尤其是在瑜伽、健身等场景下多遮挡、非常见、多卧姿姿态上具有SOTA的检测精度。

![](assets/1.png)



## 使用方式和范围
使用方式：
- 直接推理，在任意真实人物图像上进行直接推理。

使用范围:
- 包含人体的人像照片，人体分辨率大于100x100，总体图像分辨率小于1080x720，图像大小不超过2M。

目标场景:
1. 动作计数：可用于AI体测场景。
2. 动作匹配打分：可用于娱乐、健身场景等场景中，实现AI动作比对与纠错。
3. 人体动作识别：可用于监控、医疗健康等场景，通过2D人体姿态分析人体行为参数。
4. 虚拟驱动：基于2D人体姿态驱动2D/3D虚拟形象，实现低成本UCG内容生成。

### 如何使用

在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来完成人体关键点检测任务。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_hrnetv2w32_body-2d-keypoints_image'
body_2d_keypoints = pipeline(Tasks.body_2d_keypoints, model=model_id)
output = body_2d_keypoints('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/keypoints_detect/000000438862.jpg')

# the output contains poses, scores and boxes
print(output)
```

### 模型局限性以及可能的偏差

- 输入图像存在人体严重残缺的情形下，模型会出现人体或点位误检和漏检的现象。

- 高度运动模糊的情形下，模型会出现人体或点位误检和漏检的现象。
  
- 模型训练时采用帧间稳定性优化，但在视频数据上，仍然建议采用One-Euro-Filter进行帧间点位平滑后处理策略。


## 训练数据介绍
训练数据包含公开数据集（COCO，MPII， AI Challenger等）、自采人体健身图像视频、互联网搜集舞蹈图像视频等并进行标注作为训练数据。

## 模型推理流程
该模型采用自顶向下的人体关键点检测流程，模型推理分为人体检测和关键点检测两个步骤。
### 人体检测
#### 预处理
- 为了控制推理效率，图像resize到640x640分辨率输入模型。
#### 推理
- 端到端模型推理，输出备选人体检测框和score。
#### 后处理
- NMS，非最大值抑制。

### 人体关键点检测
#### 预处理
- 基于人体检测框从图像中裁剪人体，变换得到192x256大小的模型做为关键点检测模型的输入数据。
#### 推理
- 端到端模型推理，输出15点人体关键点heatmap。
#### 后处理
- 获取heatmap极大值坐标，辅以第二极大值坐标微调，获得人体关键点坐标值。
- 将人体图像空间中的关键点坐标变换到原图坐标空间中。

## 数据评估及结果
### 测评指标
COCO数据集上模型指标：
| Method | 输入大小 | AP | AP50 | AP75 | AR | AR50 |
| ------------ | ------------ | ------------ | ------------ | ------------ |------------ |------------ |
| SimpleBaseline2D | 256x192 | 0.717 | 0.898 | 0.793 | 0.772|0.936|
| HRNet | 256x192 | 0.746 | 0.904 | 0.819 | 0.799 |0.942|
| HRformer | 256x192 | 0.738 | 0.904 | 0.811 | 0.792 |0.941|
| **Ours** | 256x192 | **0.770** | 0.838 | 0.741 |  0.797 |**0.943**|

自研数据集上模型指标

| 输入大小 | PCK | 
| ------------ | ------------ | 
| 128x128 | **0.3387** |

| Head | Shoulder | Elbow | Wrist | Hip | Knee |Ankle|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |------------ |
| 0.288 | 0.275 | 0.330 | 0.400 | 0.355 | 0.350 | 0.388 |


| mAP@0.5 | mAP@0.6 | mAP@0.7 | mAP@0.8 | mAP@0.9 | mAP@0.95 |
| ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
| 84.781 | 79.093 | 68.739 | 50.955 | 22.579 | 11.956 |

### 模型效果
![人体关键检测结果](assets/results1.jpg)
![人体关键检测结果](assets/results2.jpg)
![人体关键检测结果](assets/results3.jpg)
![人体关键检测结果](assets/results5.jpg)

### 引用
```BibTeX
@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={TPAMI},
  year={2019}
}
```