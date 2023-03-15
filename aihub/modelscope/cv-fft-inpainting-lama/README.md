
# LaMa image inpainting 图像填充介绍

本模型选自LaMa算法，同时支持高分辨率图像(~2k)在线refinement，对图片进行修复，填充和编辑等。

<img src="data/1.gif" width=70% />

<img src="data/2.gif" width=70% />

## 模型描述

LaMa 采用FFT卷积+普通卷积的方式从而有效地进行图像填充，仅在256x256分辨率图像上训练，就能实现高分辨清晰图像(~2k)的填充，同时采用现在refinement策略，进一步提升高分辨率图像的填充效果

## 期望模型使用方式以及适用范围

本模型适用范围为室外自然场景；

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例


- 普通推理(支持CPU/GPU)：

可对高分辨率图像(~2k)进行图像填充，修复等。右侧demo采用该普通推理模式以减少资源消耗，若想采用精细推理请使用code进行调用

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting.png'
input_mask_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting_mask.png'
input = {
        'img':input_location,
        'mask':input_mask_location,
}

inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama')
result = inpainting(input)
vis_img = result[OutputKeys.OUTPUT_IMG]
cv2.imwrite('result.png', vis_img)
```

- 精细推理[[推荐方式]]() (仅支持GPU):

可对高分辨率图像(~2k)进行精细的图像填充，修复等，获得更加逼真的修复图片。

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting.png'
input_mask_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting_mask.png'
input = {
        'img':input_location,
        'mask':input_mask_location,
}

inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama', refine=True)
result = inpainting(input)
vis_img = result[OutputKeys.OUTPUT_IMG]
cv2.imwrite('result.png', vis_img)
```

- 模型训练：

```python
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.cv.image_inpainting import FFTInpainting
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level



model_id = 'damo/cv_fft_inpainting_lama'
cache_path = snapshot_download(model_id)
cfg = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))

train_data_cfg = ConfigDict(
    name='PlacesToydataset',
    split='train',
    mask_gen_kwargs=cfg.dataset.mask_gen_kwargs,
    out_size=cfg.dataset.train_out_size,
    test_mode=False)

test_data_cfg = ConfigDict(
    name='PlacesToydataset',
    split='test',
    mask_gen_kwargs=cfg.dataset.mask_gen_kwargs,
    out_size=cfg.dataset.val_out_size,
    test_mode=True)

train_dataset = MsDataset.load(
    dataset_name=train_data_cfg.name,
    split=train_data_cfg.split,
    mask_gen_kwargs=train_data_cfg.mask_gen_kwargs,
    out_size=train_data_cfg.out_size,
    test_mode=train_data_cfg.test_mode)
assert next(
    iter(train_dataset.config_kwargs['split_config'].values()))

test_dataset = MsDataset.load(
    dataset_name=test_data_cfg.name,
    split=test_data_cfg.split,
    mask_gen_kwargs=test_data_cfg.mask_gen_kwargs,
    out_size=test_data_cfg.out_size,
    test_mode=test_data_cfg.test_mode)
assert next(
    iter(test_dataset.config_kwargs['split_config'].values()))

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=test_dataset)

trainer = build_trainer(
    name=Trainers.image_inpainting, default_args=kwargs)
trainer.train()

```

上述训练代码仅仅提供简单训练的范例，对大规模数据，例如Places2可以进行数据替换，直接放置在对应cache中即可；此外configuration.json(~/.cache/modelscope/hub/damo/cv_fft_inpainting_lama/)可以进行自定义修改；

### 模型局限性以及可能的偏差

- 人脸修复、填充图片暂不支持
- 当前版本在python 3.8环境测试通过，其他环境下可用性待测试
- 当前版本fine-tune在cpu和单机单gpu环境测试通过，单机多gpu等其他环境待测试

## 训练数据介绍

- [Places2](http://places2.csail.mit.edu/download.html)：包括8M自然图像用于训练，30k图片用于测试和评估


## 模型训练流程

- 在Places2上使用Adam优化器，初始学习率为1e-3，训练1M iterations。

### 预处理

测试时主要的预处理如下：
- Pad：图像高宽镜像填充至8的倍数
- Normalize：图像归一化，像素大小由0-255归一化到0-1即可

## 数据评估及结果

采用我们提供的PlacesToydataset数据进行finetune后得到的结果FID一般为 ***30-80*** 之间（由于我们提供的数据少，而FID的计算依赖大量数据，故此处FID结果偏高且不稳定）

PlacesToydataset:
| models |  Pretrain   | FID |
|:--------:|:-----------:|:-------:|
|  big-lama  | ImageNet-1k |  30-80  |


注：LaMa官方模型在Places2上report结果如下:

Places val:
| models |  Pretrain   | FID |
|:--------:|:-----------:|:-------:|
|  big-lama  | ImageNet-1k |  2.97  |



## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```
```BibTeX
@article{kulshreshtha2022feature,
  title={Feature Refinement to Improve High Resolution Image Inpainting},
  author={Kulshreshtha, Prakhar and Pugh, Brian and Jiddi, Salma},
  journal={arXiv preprint arXiv:2206.13644},
  year={2022}
}
```
