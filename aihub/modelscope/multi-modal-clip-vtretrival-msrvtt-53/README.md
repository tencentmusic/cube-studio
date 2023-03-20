

# 视频-文本检索模型介绍
该模型是对视频-文本pair进行特征提取和匹配模型。输入任意视频和文本pair，输出相应的视频-文本pair特征，和相应得分。

# 数据集说明
该模型采用Howto100M的视频-文本数据集中预训练CLIP模型，然后在msrvtt数据集进行finetune。

# 模型结构
![模型结构](resources/model.png)

CLIP模型：视觉encoder采用vit-large-patch16结构，文本encoder采用bert-base结构。
Interaction: 采用weighted token-wise interaction。如上图所示。

## 模型训练
### finetune LR scheduler
初始LR为 0.0001，共训练5个epoch。

# 使用方式和范围
使用方式：
- 直接推理，对输入的视频-文本pair直接进行推理。
使用场景:
- 适合任意视频-文本pair，一般文本长度最长编码不超过77，视频时间15s-5min。

# 结果说明
MSRVTT test，R@1:53%，达到sota结果。

## 代码范例:

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_multi_modal= pipeline(
            Tasks.video_multi_modal_embedding,
            model='damo/multi_modal_clip_vtretrival_msrvtt_53')
video_path = 'your video path.mp4'
caption = ('your text caption', None, None)
_input = {'video': video_path, 'text': caption}
result = video_multi_modal(_input)
```
