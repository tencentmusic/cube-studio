

# 基础视觉模型高效调优：Adapter

基于大规模预训练基础模型的参数高效迁移学习方法在各种下游应用中均取得了优异的表现，其中包括了利用Adapter进行调优的方法。该方法通过额外添加了类bottleneck的神经网络模块，仅需训练极少部分的参数，就能取得不错的性能表现。

该页面展示了Adapter在图像分类任务上的应用，即给定一张图片，返回候选类别中的分类标签及置信度。


## 模型描述
Adapter的模型结构如下所示，其中左侧为Adapter嵌入到Vision Transformer中的框架，右侧为Adapter的具体结构：

<img src="./description/adapter_architecture.png" alt="architecture" width="40%" height="40%">

## 期望模型使用方式以及适用范围

### 如何使用

基于 ModelScope 框架，通过调用预定义的 Pipeline 可实现快速调用。


#### 代码范例

```python
from modelscope.pipelines import pipeline

adapter_pipeline = pipeline('vision-efficient-tuning',
                            'damo/cv_vitb16_classification_vision-efficient-tuning-adapter',
                             model_revision='v1.0.1')
result = adapter_pipeline('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/vision_efficient_tuning_test_1.png')
print(f'Output: {result}.')            
```


### 模型局限性以及可能的偏差

- 本模型基于公开的[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)通用数据集训练，且仅适用于训练数据的覆盖类别，在具体应用场景下可能存在偏差。
- 本模型当前仅用于图像分类任务，同时该方法可用于其他模态输入（如文本、视频等）和其他视觉下游任务（如检测、分割等）。

## 训练数据介绍

1. [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) 通用图像分类数据集，包含100个类别。
2. [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011) 鸟类细粒度分类数据集，包含200个类别。
3. [NABirds](https://dl.allaboutbirds.org/nabirds) 鸟类细粒度分类数据集，包含555个类别。
4. [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) 花卉细粒度分类数据集，包含102个类别。
5. [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) 车辆细粒度分类数据集，包含196个类别。
6. [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) 犬类细粒度分类数据集，包含120个类别。


## 数据评估及结果

模型分别在不同的预训练模型和图像分类数据集下进行评估，结果如下：

|     Dataset     | ViT-B/16 (IN-21K) | ViT-L/14 (CLIP) |
|:---------------:|:-----------------:|:---------------:|
|     CIFAR100    |       91.88%      |      91.76%     |
|   CUB-200-2011  |       87.66%      |      88.00%     |
|     NABirds     |       81.79%      |      83.47%     |
| Oxford Flowers  |       98.63%      |      98.83%     |
|  Stanford Cars  |       80.98%      |      93.11%     |
|  Stanford Dogs  |       89.00%      |      87.56%     |
|     Average     |       88.32%      |      90.46%     |

其中，ViT-B/16模型使用 [ImageNet-21K](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) 作为预训练模型，ViT-L/14使用 [CLIP](https://github.com/openai/CLIP) 作为预训练模型。

### 模型训练




## 相关论文以及引用信息


如果该模型对您有所帮助，请引用下面的相关的论文：

```BibTeX
@inproceedings{houlsby2019adapter,
  title={Parameter-Efficient Transfer Learning for {NLP}},
  author={Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Bruna and De Laroussilhe, Quentin and Gesmundo, Andrea and Attariyan, Mona and Gelly, Sylvain},
  booktitle=ICML,
  pages={2790--2799},
  year={2019},
}
```