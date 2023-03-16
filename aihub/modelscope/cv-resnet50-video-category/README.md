
# 视频分类介绍

本模型是对短视频进行内容分类，输入视频片段，输出视频内容分类，目前已经覆盖了23个一级类目/160个二级类目。

## 模型描述
模型采用resnet50网络结构提取视觉特征，采用NextVLAD网络对连续视频帧进行特征聚合。

## 使用方式和范围

使用方式：
- 直接推理，对输入的视频片段，输入视频url直接进行推理。

使用场景:
- 适合主题明确的短视频，视频不超过30秒。

### 如何使用

提供输入视频，即可以通过简单的Pipeline调用来识别结果。

#### 代码范例

```python
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.system('wget -O test.mp4 https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_category_test_video.mp4')
category_pipeline = pipeline(
            Tasks.video_category, model='damo/cv_resnet50_video-category')
result = category_pipeline('test.mp4')
print('Result: {}'.format(result))
```

### 模型局限性以及可能的偏差

- 对于快速运动，画面模糊较大的场景可能会产生误识别的现象。
- 对于主题内容不够突出的场景，可能会产生有歧义的结果。


## 训练数据
训练数据共约40w带内容分类标签的视频片段。

## 模型训练
使用128的batch size训练120 epochs. 基准学习率为0.1，训练过程中先采用线性预热策略，然后采用cosine的退火策略降低学习率。优化器采用SGD，weight decay和momentum分别为0.005和0.9。

### 预处理
-- 训练：采样16张图像，random crop、random flip
-- 测试：采样16张图像，resize短边到256，然后CenterCrop到224\*224，推理后输出。

## 数据评估及结果
通过收集线上的实际应用数据进行评测，一级类目top1精度81.38%, 二级类目top1精度66.82%。
