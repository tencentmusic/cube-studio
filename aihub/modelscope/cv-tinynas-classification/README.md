


# Tiny-NAS 介绍
Tiny-NAS 是一个高性能的神经结构搜索（NAS）框架，用于在GPU和移动设备上自动设计具有高预测精度和高推理速度的深度神经网络。
Zen-NAS 是一种典型的 Tiny-NAS 方法，它基于自主设计的 Zen-Score 来对网络结构进行打分与排序，从而搜索最优网络结构。ZenNet 是基于 Zen-NAS 设计出的高效网络结构。
ZenNet 在ImageNet 数据集上的 top-1 accuracy 指标精度与 EfficientNet-B5 (~83.6%) 相当，但是推理速度更快（4.9x times faster on V100, 10x times faster on NVIDIA T4, 1.6x times faster on Google Pixel2）。

![zen-net](misc/ZenNet_speed.png)


## 模型描述
ZenNet 是基于 Tiny-NAS (Zen-NAS) 算法设计出的高效的卷积网络结构。
本 demo 只提供 zennet\_imagenet1k\_latency12ms\_res22 backbone，其它网络结构可以从README 中获取。

## 模型使用方式以及适用范围
使用方式：
- 直接推理，在imagenet-1k 数据集上进行直接推理;
- 微调，在已经公开的模型在新数据、新任务上进行微调。


使用范围:
- 适合2D image 输入的任务。

目标场景:
- 在直接推理时，适合imagenet-1k 支持的标签集上进行直接推理;
- 在微调场景下，本模型可以作为各种使用 2D image下游任务的 backbone（诸如检测、分割等）。



### 如何使用
作为通用 backbone 可以被集成到各种任务中作为特征提取器。


#### 代码范例
```python

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

tinynas_classification = pipeline(
        Tasks.image_classification,
        model='damo/cv_tinynas_classification')
result = tinynas_classification('data/test/images/image_wolf.jpeg')
print(f'recognition output: {result}.')

```

### 模型局限性以及可能的偏差
考虑 GPU 精度等的差异，可能带来一定的性能差异

## 训练数据介绍
使用标准 imagenet-1k 数据集


## 模型训练流程

### 预处理
训练时：随机裁剪一部分图片，然后 resize 为 224x224，再随机翻转和颜色抖动
推理时：进行中心裁剪


### 训练
初始 learning rate 为0.4，衰减方式为cosine，weight decay 为0.00004，warmup-epochs 为5，一共迭代 200 epoch

## 数据评估及结果

| model | resolution | \# params | FLOPs | Top-1 Acc | V100 | T4 | Pixel2 |
| ----- | ---------- | -------- | ----- | --------- | ---- | --- | ------ |
| [zennet\_imagenet1k\_flops400M\_SE\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_flops400M_SE_res224/student_best-params_rank0.pth) | 224 | 5.7M | 410M | 78.0% | 0.25 | 0.39 | 87.9 |
| [zennet\_imagenet1k\_flops600M\_SE\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_flops600M_SE_res224/student_best-params_rank0.pth) | 224 | 7.1M | 611M | 79.1% | 0.36 | 0.52 | 128.6 |
| [zennet\_imagenet1k\_flops900M\_SE\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/zennet_imagenet1k_flops900M_SE_res224/student_best-params_rank0.pth) | 224 | 19.4M | 934M | 80.8% | 0.55 | 0.55 | 215.7 |
| [zennet\_imagenet1k\_latency01ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency01ms_res224/student_best-params_rank0.pth) | 224 | 30.1M | 1.7B | 77.8% | 0.1 | 0.08 | 181.7 |
| [zennet\_imagenet1k\_latency02ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency02ms_res224/student_best-params_rank0.pth) | 224 | 49.7M | 3.4B | 80.8% | 0.2 | 0.15 | 357.4 |
| [zennet\_imagenet1k\_latency03ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency03ms_res224/student_best-params_rank0.pth) | 224 | 85.4M | 4.8B | 81.5% | 0.3 | 0.20 | 517.0 |
| [zennet\_imagenet1k\_latency05ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency05ms_res224/student_best-params_rank0.pth) | 224 | 118M | 8.3B | 82.7% | 0.5 | 0.30 | 798.7 |
| [zennet\_imagenet1k\_latency08ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency08ms_res224/student_best-params_rank0.pth) | 224 | 183M | 13.9B | 83.0% | 0.8 | 0.57 | 1365 |
| [zennet\_imagenet1k\_latency12ms\_res224](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/ZenNet/pretrained_models/iccv2021_zennet_imagenet1k_latency12ms_res224/student_best-params_rank0.pth) | 224 | 180M | 22.0B | 83.6% | 1.2 | 0.85 | 2051 |
| EfficientNet-B3 | 300 | 12.0M | 1.8B | 81.1% | 1.12 | 1.86 | 569.3 |
| EfficientNet-B5 | 456 | 30.0M | 9.9B | 83.3% | 4.5 | 7.0 | 2580 |
| EfficientNet-B6 | 528 | 43M | 19.0B | 84.0% | 7.64 | 12.3 | 4288 |

* 'V100' is the inference latency on NVIDIA V100 in milliseconds, benchmarked at batch size 64, float16.
* 'T4' is the inference latency on NVIDIA T4 in milliseconds, benchmarked at batch size 64, TensorRT INT8.
* 'Pixel2' is the inference latency on Google Pixel2 in milliseconds, benchmarked at single image.


### 相关论文以及引用信息
```BibTeX

@inproceedings{ming_zennas_iccv2021,
  author    = {Ming Lin and Pichao Wang and Zhenhong Sun and Hesen Chen and Xiuyu Sun and Qi Qian and Hao Li and Rong Jin},
  title     = {Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition},
  booktitle = {2021 IEEE/CVF International Conference on Computer Vision, {ICCV} 2021},  
  year      = {2021},
}
```
If you are interested in ZenNAS, welcome to our [github](https://github.com/idstcv/ZenNAS)!
