
# 图像分割-商品展示图场景的商品分割-电商领域


这是一个通用商品的分割模型，输入一个商品宣传图，输出分割结果

## 模型描述

模型结构基于[F3Net](https://github.com/weijun88/F3Net)，同时优化了Loss的方式

## 使用方式和范围


### 如何使用

在ModelScope框架上，提供商品图片，得到分割的结果

#### 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

product_segmentation = pipeline(Tasks.product_segmentation, model='damo/cv_F3Net_product-segmentation')
result_status = product_segmentation({'input_path': 'data/test/images/product_segmentation.jpg'})
result = result_status[OutputKeys.MASKS]

```

input_path为输入图片的路径，result为numpy格式的mask

### 模型局限性以及可能的偏差

- 适用于商品广告图、宣传图场景下的商品分割，同时如果有多个商品时分割效果会受到影响


## 训练数据介绍

训练数据来自互联网搜索的图片


## 引用
```
@inproceedings{F3Net,
  title     = {F3Net: Fusion, Feedback and Focus for Salient Object Detection},
  author    = {Jun Wei, Shuhui Wang, Qingming Huang},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2020}
}
```