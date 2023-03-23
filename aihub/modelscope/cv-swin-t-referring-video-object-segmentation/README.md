
# MTTR视频目标分割模型介绍

本模型使用RoBERTa-base和video swin transformer分别来提取输入的文本特征和视频特征，并给出视频中由此文本指定的物体mask。

本模型的生成效果如下所示，输入的指导文本为：
- 'guy in black performing tricks on a bike' ——— 使用蓝色mask
- 'a black bike used to perform tricks' ——— 使用红色mask
<p align="center">
    <br>
        <video src="https://dmshared.oss-cn-hangzhou.aliyuncs.com/shuying.sy/mttr_maas/output_clip.mp4" loop="loop" autoplay="autoplay" muted width="50%"></video>
    <br>
</p>


## 模型描述
本模型采用端到端的训练方式，具体的模型如下图所示。首先，输入的文本和视频帧通过特征编码器，并且每一帧会形成一个多模态序列。
接下来，多模态 Transformer 对特征关系进行编码，并将实例级特征解码为一组预测序列，
然后生成相应的掩码和预测的序列。
最后，在训练阶段，预测序列会与标签序列进行匹配以训练网络；在推理阶段，预测序列会用于生成最终的预测结果。


<p align="center">
    <br>
    <img src="https://modelscope.cn/api/v1/models/damo/cv_swin-t_referring_video-object-segmentation/repo?Revision=master&FilePath=data/mttr-schema.jpg&View=true" width="800" />
    <br>
<p>

## 期望模型使用方式以及适用范围

使用方式：
- 直接推理，在任意的的视频上进行推理。

适用范围：
- 本模型主要应用于视频目标分割领域，推理阶段输入视频的长度须小于等于10秒，输入文本的个数须小于等于2个。



### 如何使用
在ModelScope框架上，提供输入视频、指导文本，即可以通过简单的Pipeline调用来使用本模型。

如您期望结果进行可视化，可以将下载下来的`configuration.json`文件中`pipeline`下面的`save_masked_video`设为`true`并配置好`output_path`路径。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/referring_video_object_segmentation_test_video.mp4'
text_queries = [
    'guy in black performing tricks on a bike',
    'a black bike used to perform tricks'
]

input_tuple = (input_location, text_queries)
pp = pipeline(Tasks.referring_video_object_segmentation, model='damo/cv_swin-t_referring_video-object-segmentation')
result = pp(input_tuple)
print(result)
```
### 模型局限性和可能的偏差
由于目前提供的模型只在[Refer-YouTube-VOS](https://competitions.codalab.org/competitions/29139#participate-get_data)数据集上进行了预训练，因此，超出此数据集domain覆盖的视频或者指导文本可能会影响推理时的分割结果。




## 训练数据介绍
训练数据为[Refer-YouTube-VOS](https://competitions.codalab.org/competitions/29139#participate-get_data)公开数据集和[A2D_Sentences](https://kgavrilyuk.github.io/publication/actor_action/)公开数据集。

## 模型的训练流程
模型微调和评估流程及代码可见[文档](https://www.modelscope.cn/docs/MTTR_referring_video_object_segmentation)【模型微调示例】小节。

## 模型推理流程

### 预处理
- 将输入视频按照输入的起始时间进行裁剪
- 调整视频的分辨率为360*640
- 对视频图像进行归一化

### 推理
- 将视频帧按照一定的长度进行聚合，形成若干个互相重叠的序列
- 将每个视频帧序列与输入的每个指导文本进行结合
- 使用模型对每一个序列进行结果预测
- 生成每个指导文本对应的每一帧的mask结果

## 数据评估及结果
| DataSet            | mAP  | J&F   |
|:------------------:|:----:|:-----:|
| AD-Sentences       | 46.1 | -     |
| JHMDB-Sentences    | 39.2 | -     |
| Refer-YouTube-VOS  | -    | 55.32 |


### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```BibTeX
@inproceedings{botach2021end,
  title={End-to-End Referring Video Object Segmentation with Multimodal Transformers},
  author={Botach, Adam and Zheltonozhskii, Evgenii and Baskin, Chaim},
  booktitle={Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```