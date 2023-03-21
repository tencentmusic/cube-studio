# 全身133点关键点检测模型

输入一张人物图像，端到端检测全身133点关键点，输出人体框和对应的全身关键点，包含68个人脸关键点、42个手势关键点、17个骨骼关键点和6个脚部关键点。

## 133点人体关键点
![全身关键点定义](assets/keypoints.png)

## 模型描述
该任务采用自顶向下的全身关键点检测框架(如下图)，通过端对端的快速推理可以得到图像中的人体关键点。其中全身关键点模型基于HRNet的backbone，充分利用多分变率的特征融合，良好支持日常人体姿态，具有SOTA的检测精度。

![](assets/1.png)

## 使用方式和范围
使用方式：
- 直接推理，在任意真实人物图像上进行直接推理;

使用范围:
- 包含人体的图片，人体分辨率大于100x100，总体图像分辨率小于1080x720，图像大小不超过2M。

目标场景:
1. 动作计数
2. 动作匹配打分
3. 人体动作识别
4. 虚拟驱动
5. 手势识别
6. 人脸识别

### 如何使用

在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来完成人体关键点检测任务。

#### 代码范例
```python
# numpy >= 1.20
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_hrnetw48_human-wholebody-keypoint_image'
wholebody_2d_keypoints = pipeline(Tasks.human_wholebody_keypoint, model=model_id)
output = wholebody_2d_keypoints('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/keypoints_detect/img_test_wholebody.jpg')

# the output contains keypoints and boxes
print(output)
```

### 模型局限性以及可能的偏差

- 输入图像存在人体严重残缺或遮挡的情形下，模型会出现人体或点位误检和漏检的现象。

- 高度运动模糊的情形下，模型会出现人体或点位误检和漏检的现象。
  
- 模型训练时采用帧间稳定性优化，但在视频数据上，仍然建议采用One-Euro-Filter进行帧间点位平滑后处理。


## 训练数据介绍
训练数据使用COCO公开数据集（https://cocodataset.org/#download）。

## 模型推理流程
该模型采用自顶向下的人体关键点检测流程，模型推理分为人体检测和关键点检测两个步骤。
### 人体检测
#### 推理
- 端到端模型推理，输出备选人体检测框和score。

### 人体关键点检测
#### 预处理
- 基于人体检测框从图像中裁剪人体，变换得到384x288大小的模型输入数据。
#### 推理
- 端到端模型推理，输出133点人体关键点heatmap。
#### 后处理
- 获取heatmap极大值坐标，辅以第二极大值坐标微调，获得人体关键点坐标值。
- 将人体图像空间中的关键点坐标变换到原图坐标空间中。

## 数据评估及结果
### 测评指标
COCO数据集上模型指标：
| Method | 输入大小 | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR |
| ------------ | ------------ | ------------ | ------------ | ------------ |------------ |------------ |------------ |------------ |------------ |
| **hrnetw48** | 384x288 | 0.742 | 0.807 | 0.705 |  0.804 |  0.840  |  0.892  |  0.602  |  0.694  |


### 模型效果
![人体关键检测结果](https://modelscope.cn/api/v1/models/damo/cv_hrnetw48_human-wholebody-keypoint_image/repo?Revision=master&FilePath=assets/000000567740_ret.jpg&View=true)
![人体关键检测结果](https://modelscope.cn/api/v1/models/damo/cv_hrnetw48_human-wholebody-keypoint_image/repo?Revision=master&FilePath=assets/000000053626_ret.jpg&View=true)
![人体关键检测结果](https://modelscope.cn/api/v1/models/damo/cv_hrnetw48_human-wholebody-keypoint_image/repo?Revision=master&FilePath=assets/000000000785_ret.jpg&View=true)
![人体关键检测结果](https://modelscope.cn/api/v1/models/damo/cv_hrnetw48_human-wholebody-keypoint_image/repo?Revision=master&FilePath=assets/000000566282_ret.jpg&View=true)

### 引用
```BibTeX
@InProceedings{Sun_2019_CVPR,
author = {Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
title = {Deep High-Resolution Representation Learning for Human Pose Estimation},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
