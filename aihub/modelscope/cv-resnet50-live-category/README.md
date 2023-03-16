
# 直播类目介绍

本模型是对直播视频进行商品类目识别，输入视频片段，输出直播商品类目标签，目前已经覆盖了8K多类的细粒度的商品类别。

## 模型描述
模型采用resnet50网络结构。

## 使用方式和范围

使用方式：
- 直接推理，对输入的视频片段，输入视频url直接进行推理。

使用场景:
- 适合含有商品的直播间口播视频，期望画面中商品占比不要过小，视频不超过30秒。

### 如何使用

提供输入视频，即可以通过简单的Pipeline调用来识别结果。

#### 代码范例

```python
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.system('wget -O test.mp4 https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/live_category_test_video.mp4')
category_pipeline = pipeline(
            Tasks.live_category, model='damo/cv_resnet50_live-category')
result = category_pipeline('test.mp4')
print('Top-3 result: {}'.format(result))
```

### 模型局限性以及可能的偏差

- 在背景环境复杂，干扰物体较多的场景下，模型可能会出现误识别的现象。
- 训练数据以直播场景为主，其他通用场景下可能不适用，具体效果需要用户自行评估。


## 训练数据
训练数据共约400w带有商品类目标签的直播片段。

## 模型训练
使用1024的batch size训练120 epochs. 基准学习率为0.4，训练过程中先采用线性预热策略，然后采用cosine的退火策略降低学习率。优化器采用SGD，weight decay和momentum分别为0.005和0.9。

### 预处理
-- 训练：random crop、random flip
-- 测试：采样4张图像，resize到256\*256，然后CenterCrop到224\*224，推理结果平均后输出。

## 数据评估及结果
通过收集线上的实际应用数据进行评测精度为71.3%。
