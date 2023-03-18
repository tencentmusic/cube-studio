
# Cascade-Mask-RCNN-Swin实例分割模型介绍

本模型基于Cascade mask rcnn分割框架，backbone选用先进的swin transformer模型。

## 模型描述

Swin transformer是一种具有金字塔结构的transformer架构，其表征通过shifted windows计算。Shifted windows方案将自注意力的计算限制在不重叠的局部窗口上，同时还允许跨窗口连接，从而带来更高的计算效率。分层的金字塔架构则让其具有在各种尺度上建模的灵活性。这些特性使swin transformer与广泛的视觉任务兼容，并在密集预测任务如COCO实例分割上达到SOTA性能。其结构如下图所示。

![Swin模型结构](description/teaser.png)

Cascade R-CNN是一种多阶段目标检测架构，该架构由一系列经过不断提高的IOU阈值的检测器组成。检测器串联进行训练，前一个检测器的输出作为下一个检测器的输入。通过重采样不断提高proposal质量，达到高质量检测定位的效果。Cascade R-CNN可以被推广到实例分割，并对Mask R-CNN产生重大改进。其结构示意图如下所示。

![Cascade-rcnn模型结构](description/cascade-rcnn.jpg)

## 期望模型使用方式以及适用范围

本模型适用范围较广，能对图片中包含的大部分感兴趣物体（COCO 80类）进行识别和分割。

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks

input_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_instance_segmentation.jpg'
output = './result.jpg'
segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_swin-b_image-instance-segmentation_coco')
result = segmentation_pipeline(input_img)

# if you want to show the result, you can run
from modelscope.preprocessors.image import LoadImage
from modelscope.models.cv.image_instance_segmentation.postprocess_utils import show_result

numpy_image = LoadImage.convert_to_ndarray(input_img)[:, :, ::-1]   # in bgr order
show_result(numpy_image, result, out_file=output, show_box=True, show_label=True, show_score=False)

from PIL import Image
Image.open(output).show()
```

### 模型局限性以及可能的偏差

- 部分非常规图片或感兴趣物体占比太小或遮挡严重可能会影响分割结果
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试
- 当前版本fine-tune在cpu和单机单gpu环境测试通过，单机多gpu等其他环境待测试

## 训练数据介绍

- [COCO2017](https://cocodataset.org/#detection-2017)：COCO全称是Common Objects in Context，是Microsoft团队提供的一个可以用来图像识别、检测和分割的数据集。COCO2017包含训练集118287张、验证集5000张、测试集40670张，共有80类物体。


## 模型训练流程

- 在COCO上使用AdamW优化器，初始学习率为1e-4；训练过程中使用large scale jitter和simple copy paste数据增强，训练了更长时间（50 epoch）；Swin transformer使用ImageNet-1K上的预训练模型

### 预处理

测试时主要的预处理如下：
- Resize：先将原始图片的短边Resize到800，等比例缩放。此时如果长边超过了1333，则按照最长边为1333，重新计算Resize的scale进行Resize
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

## 数据评估及结果

模型在COCO2017val上进行测试，结果如下:

| Backbone |  Pretrain   | box mAP | mask mAP | #params | FLOPs  | Remark       | 
|:--------:|:-----------:|:-------:|:--------:|:-------:|:------:|--------------|
|  Swin-B  | ImageNet-1k |  51.9   |   45.0   |  145M   |  982G  | [official](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) |
|  Swin-B  | ImageNet-1k |  52.7   |   46.1   |  145M   |  982G  | modelscope   |

可视化结果：

![source](description/demo.jpg)  ![result](description/result.jpg)


### 模型评估代码
可通过如下代码对模型进行评估验证，我们在modelscope的[DatasetHub](https://modelscope.cn/datasets/modelscope/COCO2017_Instance_Segmentation/summary)上存储了COCO2017的验证集，方便用户下载调用。
```python
from functools import partial
import os
import tempfile

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode

from mmcv.parallel import collate


tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

eval_dataset = MsDataset.load('COCO2017_Instance_Segmentation', split='validation',  
                              download_mode=DownloadMode.FORCE_REDOWNLOAD)
kwargs = dict(
    model='damo/cv_swin-b_image-instance-segmentation_coco',
    data_collator=partial(collate, samples_per_gpu=1),
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)
trainer = build_trainer(name=Trainers.image_instance_segmentation, default_args=kwargs)
metric_values = trainer.evaluate()
print(metric_values)
```

### 模型训练代码
通过使用托管在modelscope DatasetHub上的数据集（持续更新中）：
```python
from functools import partial

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.hub import read_config


WORKSPACE = './work_dir'
model_id = 'damo/cv_swin-b_image-instance-segmentation_coco'

samples_per_gpu = read_config(model_id).train.dataloader.batch_size_per_gpu
train_dataset = MsDataset.load(dataset_name='pets_small',split='train')
eval_dataset = MsDataset.load(dataset_name='pets_small', split='validation', test_mode=True)
max_epochs = 1

from mmcv.parallel import collate
    
kwargs = dict(
            model=model_id,
            data_collator=partial(collate, samples_per_gpu=samples_per_gpu),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=WORKSPACE,
            max_epochs=max_epochs)

trainer = build_trainer(
    name=Trainers.image_instance_segmentation, default_args=kwargs)

print('===============================================================')
print('pre-trained model loaded, training started:')
print('===============================================================')

trainer.train()

print('===============================================================')
print('train success.')
print('===============================================================')

for i in range(max_epochs):
    eval_results = trainer.evaluate(f'{WORKSPACE}/epoch_{i+1}.pth')
    print(f'epoch {i} evaluation result:')
    print(eval_results)


print('===============================================================')
print('evaluate success')
print('===============================================================')
```

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
```BibTeX
@article{Cai_2019,
   title={Cascade R-CNN: High Quality Object Detection and Instance Segmentation},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/tpami.2019.2956516},
   DOI={10.1109/tpami.2019.2956516},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cai, Zhaowei and Vasconcelos, Nuno},
   year={2019},
   pages={1–1}
}
```
