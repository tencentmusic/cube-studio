# Multi-dimension Attention Network for Image Quality Assessment

## 模型描述
MAN模型主要用于无参考图像质量评估。模型首先通过ViT提取特征，然后为了加强全局和局部交互，使用了转置注意力模块(TAB)和Swin Transformer模块(SSTB)。这两个模块分别在通道和空间维度上应用注意力机制。以这种多维方式，这些模块合作地增加了图像不同区域之间的全局和局部交互。最后模型使用基于Patch权重的质量预测的双分支结构来预测最终得分。该模型在 NTIRE2022 图像视觉质量评价Challenge，无参考赛道取得了第一名。

## 期望模型使用方式以及适用范围
本模型适用于图像的视觉质量评价，输出评价mos分,范围[0, 1],值越大代表图像质量越好。模型适用于224x224的图像,如果图像分辩率较大,建议采用多个位置Crop取平均。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img = 'test.jpg'
image_quality_assessment_pipeline = pipeline(Tasks.image_quality_assessment_mos, 'damo/cv_man_image-quality-assessment')
result = image_quality_assessment_pipeline(img)[OutputKeys.SCORE]
print(result)
```

### 模型局限性以及可能的偏差
由于训练数据为PIPAL2022，对其他类型图像可能表现不佳。


### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@inproceedings{yang2022maniqa,
  title={MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment},
  author={Yang, Sidi and Wu, Tianhe and Shi, Shuwei and Lao, Shanshan and Gong, Yuan and Cao, Mingdeng and Wang, Jiahao and Yang, Yujiu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1191--1200},
  year={2022}
}
```
