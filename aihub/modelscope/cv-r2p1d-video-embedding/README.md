
# CMD视频自监督学习方法介绍


## 模型描述
针对视频理解领域的「场景偏差」难题（例如：一段「在篮球场跳舞」的视频，会被识别为打篮球，而非跳舞），我们提出一种自监督视频表征学习方案，通过在代理任务（pretext tasks）中显式地解耦场景与运动信息（context and motion decoupling），强制视频模型同时捕捉静态背景与动态行为两方面特征。值得注意的是，本方案中，解耦的场景与运动数据均从「视频压缩编码」(例如：H.264) 中提取得到，其中场景由关键帧 (keyframes) 表示，运动由运动向量 (motion vectors) 表示，二者在CPU上的提取速度达500 fps，是光流 (另一种运动表示) 的100倍。基于该解藕方案预训练的视频网络模型，迁移至行为理解和视频检索两项下游任务，性能均显著超过SOTA。

其模型结构如下所示：

![模型结构](description/framework.png)


## 使用方式和范围

使用方式：
- 直接推理提取视频特征

使用范围:
- 视频分辨率在112x112以上，输入片段宜在10s以内

目标场景:
- 体育、影视、直播等视频特征抽取，用于下游的视频检索、视频分类任务。


### 如何使用

提供输入视频，即可以通过简单的Pipeline调用来提取视频特征向量。


#### 代码范例
```python
import os
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.system('wget -O test.mp4 https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/action_recognition_test_video.mp4')
videossl_pipeline = pipeline(Tasks.video_embedding, 'damo/cv_r2p1d_video_embedding')
result = videossl_pipeline('test.mp4')

print(f'video embedding: {result}.')
```

### 模型局限性以及可能的偏差

- 考虑GPU精度、视频解码工具的差异，可能带来一定的性能差异，请用户自行评测后决定如何使用。
- 在CPU测试速度比V100 GPU测试速度相差约3倍。
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试。

## 训练数据介绍

- [Kinetics-400](https://www.deepmind.com/open-source/kinetics) 常用行为识别的公开数据集，包含400类，总共有接近30万视频；

- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) 包含13,320视频共计101种动作分类；

- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#dataset) 包含6,766视频片段共计51种动作分类。


## 模型训练流程
- 预训练（论文）：在UCF101数据集上，使用64的batch size训练120 epochs. 在Kinetics400数据集上，以512的batch size训练120 epochs. 基准学习率为0.0005xB，其中B为batch size大小，训练过程中先采用线性预热策略，然后采用cosine的退火策略降低学习率。优化器采用SGD，weight decay和momentum分别为0.005和0.9。

- 微调（论文）：在UCF101数据集上，基准学习率设置为0.0001xB, SGD优化器的weight decay设置为0.003；在HMDB51数据集上，基准学习率设置为0.0002xB, SGD优化器的weight decay设置为0.002。两个数据集均采用batch size为8，训练120 epochs。

- 暂时不支持通过ModelScope接口进行微调。

## 数据评估及结果

在行为识别和视频检索两个任务上进行测试:

- 行为识别，在UCF101、Kinetics400预训练，然后在UCF101、HMDB51上进行微调，统计top1 识别精度。

| Pretrained | Resolution | Architecture | UCF101 | HMDB51 | 
| ------------ | ------------ | ------------ | ------------ | ------------ |
| UCF101 | 112x112 | C3D | 78.6 | 46.9 | 
| UCF101 | 112x112 | R(2+1)D-26 | 79.7 | 48.6 | 
| UCF101 | 112x112 | R3D-26 | 76.6 | 47.2 | 
| Kinetics400 | 112x112 | C3D | 83.4 | 52.9 | 
| Kinetics400 | 112x112 | R(2+1)D-26 | 85.7 | 54.0 | 
| Kinetics400 | 112x112 | R3D-26 | 83.7 | 55.2 | 

- 视频检索, 在UCF101 预训练，然后在UCF101、HMDB51上提取特征直接进行视频检索测试，统计R@1。

| Pretrained | Resolution | Architecture | UCF101 | HMDB51 | 
| ------------ | ------------ | ------------ | ------------ | ------------ |
| UCF101 | 112x112 | C3D | 66.9 | 50.0 | 
| UCF101 | 112x112 | R(2+1)D-26 | 65.2 | 48.5 | 
| UCF101 | 112x112 | R3D-26 | 65.8 | 51.4 |


## 引用
如果你觉得这个该模型对你有所帮助，请考虑引用下面的论文：

```BibTeX
@inproceedings{huang2021self,
  title={Self-supervised video representation learning by context and motion decoupling},
  author={Huang, Lianghua and Liu, Yu and Wang, Bin and Pan, Pan and Xu, Yinghui and Jin, Rong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13886--13895},
  year={2021}
}
```