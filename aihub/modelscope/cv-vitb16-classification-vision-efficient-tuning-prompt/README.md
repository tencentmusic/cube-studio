

# 基础视觉模型高效调优：Prompt

基于大规模预训练基础模型的参数高效迁移学习方法在各种下游应用中均取得了优异的表现，其中包括了利用Prompt进行调优的方法。该方法对输入的块嵌入向量额外添加了多个可学习的prompt向量，仅需训练极少部分的参数，就能取得不错的性能表现。

该页面展示了Prompt在图像分类任务上的应用，即给定一张图片，返回候选类别中的分类标签及置信度。


## 模型描述
Prompt的模型结构如下所示，其中左侧为Prompt嵌入到Vision Transformer中的框架，右侧为Prompt的具体结构：

<img src="./description/prompt_architecture.png" alt="architecture" width="40%" height="40%">

## 期望模型使用方式以及适用范围

### 如何使用

基于 ModelScope 框架，通过调用预定义的 Pipeline 可实现快速调用。


#### 代码范例

```python
from modelscope.pipelines import pipeline

prompt_pipeline = pipeline('vision-efficient-tuning',
                            'damo/cv_vitb16_classification_vision-efficient-tuning-prompt',
                             model_revision='v1.0.1')
result = prompt_pipeline('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/vision_efficient_tuning_test_1.png')
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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="2">Dataset</th>
    <th class="tg-9wq8" colspan="2">ViT-B/16 (IN-21K)</th>
    <th class="tg-9wq8" colspan="2">ViT-L/14 (CLIP)</th>
  </tr>
  <tr>
    <th class="tg-9wq8">Prompt Shallow</th>
    <th class="tg-9wq8">Prompt Deep</th>
    <th class="tg-9wq8">Prompt Shallow</th>
    <th class="tg-9wq8">Prompt Deep</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8">CIFAR100</td>
    <td class="tg-9wq8">86.62%</td>
    <td class="tg-9wq8">91.58%</td>
    <td class="tg-9wq8">88.95%</td>
    <td class="tg-9wq8">91.28%</td>
  </tr>
  <tr>
    <td class="tg-9wq8">CUB-200-2011</td>
    <td class="tg-9wq8">86.92%</td>
    <td class="tg-9wq8">87.99%</td>
    <td class="tg-9wq8">85.64%</td>
    <td class="tg-9wq8">85.92%</td>
  </tr>
  <tr>
    <td class="tg-9wq8">NABirds </td>
    <td class="tg-9wq8">79.31%</td>
    <td class="tg-9wq8">82.77%</td>
    <td class="tg-9wq8">80.25%</td>
    <td class="tg-9wq8">82.18%</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Oxford Flowers </td>
    <td class="tg-9wq8">98.46%</td>
    <td class="tg-9wq8">98.60%</td>
    <td class="tg-9wq8">98.55%</td>
    <td class="tg-9wq8">97.72%</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Stanford Cars </td>
    <td class="tg-9wq8">55.45%</td>
    <td class="tg-9wq8">78.25%</td>
    <td class="tg-9wq8">90.65%</td>
    <td class="tg-9wq8">91.82%</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Stanford Dogs </td>
    <td class="tg-9wq8">89.53%</td>
    <td class="tg-9wq8">90.27%</td>
    <td class="tg-9wq8">84.22%</td>
    <td class="tg-9wq8">85.28%</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Average</td>
    <td class="tg-9wq8">82.72%</td>
    <td class="tg-9wq8">88.24%</td>
    <td class="tg-9wq8">88.04%</td>
    <td class="tg-9wq8">89.03%</td>
  </tr>
</tbody>
</table>

其中，ViT-B/16模型使用 [ImageNet-21K](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) 作为预训练模型，ViT-L/14使用 [CLIP](https://github.com/openai/CLIP) 作为预训练模型。


## 相关论文以及引用信息


如果该模型对您有所帮助，请引用下面的相关的论文：

```BibTeX
@inproceedings{jia2022vpt,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle=ECCV,
  year={2022}
}
```