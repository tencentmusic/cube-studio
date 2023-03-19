# OFA表情包文本生成器
## News
- 2023年2月：
  - 进一步扩大和过滤非法文本内容的，并原模型上继续迭代微调以产生更加多元化和健康的表情包文本, 并提供创空间体验不同版本效果。
- 2023年1月：
  - 预处理的4W表情包数据集，在ofa-large模型上finetune后的微调模型，根据提供的图片生产对应的表情包文本。

# 表情包文本生成(中文)
# 什么是表情包文本生成？
表情包文本生成指的是给一张图片，配一个表情包style的文字。

## 快速玩起来
玩转OFA只需区区以下6行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角Notebook按钮，我们为你提供了配备了GPU的环境，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了！

<p align="center">
    <img src="resources/mushroom_head1.png" alt="letsgo" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_meme_large_zh',model_revision='v1.0.2')
result = img_captioning('http://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/meme/mushroom_head1.png')
print(result[OutputKeys.CAPTION])  # ['我不想说话', '我是你爸爸']
```

# OFA是什么？
OFA(One-For-All)是通用多模态预训练模型，使用简单的序列到序列的学习框架统一模态（跨模态、视觉、语言等模态）和任务（如图片生成、视觉定位、图片描述、图片分类、文本生成等），详见我们发表于ICML 2022的论文：[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)，以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)。

<p align="center">
    <br>
    <img src="resources/OFA_logo_tp_path.svg" width="150" />
    <br>
<p>
<br>

<p align="center">
        <a href="https://github.com/OFA-Sys/OFA">Github</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2202.03052">Paper </a>&nbsp ｜ &nbspBlog
</p>

<p align="center">
    <br>
        <video src="https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/resources/modelscope_web/demo.mp4" loop="loop" autoplay="autoplay" muted width="100%"></video>
    <br>
</p>

# OFA模型规模

<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Params-en</th><th>Params-zh</th><th>Backbone</th><th>Hidden size</th><th>Intermediate size</th><th>Num. of heads</th><th>Enc layers</th><th>Dec layers</th>
    </tr>
    <tr align="center">
        <td>OFA<sub>Tiny</sub></td><td>33M</td><td>-</td><td>ResNet50</td><td>256</td><td>1024</td><td>4</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Medium</sub></td><td>93M</td><td>-</td><td>ResNet101</td><td>512</td></td><td>2048</td><td>8</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td>180M</td><td>160M</td><td>ResNet101</td><td>768</td></td><td>3072</td><td>12</td><td>6</td><td>6</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td>470M</td><td>440M</td><td>ResNet152</td><td>1024</td></td><td>4096</td><td>16</td><td>12</td><td>12</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Huge</sub></td><td>930M</td><td>-</td><td>ResNet152</td><td>1280</td></td><td>5120</td><td>16</td><td>24</td><td>12</td>
    </tr>
</table>
<br>

# OFA 模型任务矩阵
目前ModelScope上面所有已经上传的模型和任务可以在下面导航表格看到，点击链接可以跳转到相应modelcard。

| 模型规模 | 预训练 | 图像描述 | 视觉问答 | 视觉定位 | 视觉蕴含 | 文生图 | 图像分类 | 文字识别 | 文本摘要 | 文本分类 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OFA<sub>Tiny</sub> | [英文](https://modelscope.cn/models/damo/ofa_pretrain_tiny_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_distilled_en/summary) | - | [英文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_distilled_en) | [英文](https://modelscope.cn/models/damo/ofa_visual-entailment_snli-ve_distilled_v2_en/summary) | - | - | - | - | - |
| OFA<sub>Medium</sub> | [英文](https://modelscope.cn/models/damo/ofa_pretrain_medium_en/summary)  | - | - | - | - | - | - | - | - | - |
| OFA<sub>Base</sub> | [中文](https://modelscope.cn/models/damo/ofa_pretrain_base_zh/summary)/[英文](https://modelscope.cn/models/damo/ofa_pretrain_base_en/summary) | [中文电商](https://modelscope.cn/models/damo/ofa_image-caption_muge_base_zh/summary) | - | - | - | - | - | [场景中文](https://modelscope.cn/models/damo/ofa_ocr-recognition_scene_base_zh/summary) | - | - |
| OFA<sub>Large</sub> | [中文](https://modelscope.cn/models/damo/ofa_pretrain_large_zh/summary)/[英文](https://modelscope.cn/models/damo/ofa_pretrain_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_visual-question-answering_pretrain_large_en/summary) | [中文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_large_zh/summary)/[英文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_visual-entailment_snli-ve_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_text-to-image-synthesis_coco_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_image-classification_imagenet_large_en/summary) | - | [英文](https://modelscope.cn/models/damo/ofa_summarization_gigaword_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_text-classification_mnli_large_en/summary) |
| OFA<sub>Huge</sub> | [英文](https://modelscope.cn/models/damo/ofa_pretrain_huge_en/summary)  | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_huge_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_visual-question-answering_pretrain_huge_en/summary) | - | - | - | - | - | - | - |
| OFA<sub>6B</sub> | - | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_6b_en/summary) | - | - | - | - | - | - | - | - |


# 相关论文以及引用
如果你觉得OFA好用，喜欢我们的工作，欢迎引用：
```
@article{wang2022ofa,
  author    = {Peng Wang and
               An Yang and
               Rui Men and
               Junyang Lin and
               Shuai Bai and
               Zhikang Li and
               Jianxin Ma and
               Chang Zhou and
               Jingren Zhou and
               Hongxia Yang},
  title     = {OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence
               Learning Framework},
  journal   = {CoRR},
  volume    = {abs/2202.03052},
  year      = {2022}
}
```