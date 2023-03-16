
# NAFNet: Nonlinear Activation Free Network for Image Restoration

## 模型描述
NAFNet（Nonlinear Activation Free Network）提出了一个简单的基线，计算效率高。其不需要使用非线性激活函数（Sigmoid、ReLU、GELU、Softmax等），可以达到SOTA性能。其网络结构如下图所示：

<img src="./data/image_restoration_arch.png" height=448 alt="Image Restoration Architecture">
<br />
Figure 1: Comparison of architectures of image restoration models.
<br/>

<img src="./data/NAFNet_arch.png" height=600 alt="NAFNet architecture">
<br/>
Figure 2: Intra-block structure comparison.
<br/>
<img src="./data/attention.png" height=224 alt="attention">

<br/>

| <img src="./data/denoise.gif"  height=224 width=224 alt="NAFNet For Image Denoise">|
| :-----------------------------------------------------------------------------------: |
|                                     Denoise                                           |

## NAFNet系列模型

| [图像去模糊](https://modelscope.cn/models/damo/cv_nafnet_image-deblur_gopro/summary) | [图像去模糊压缩](https://modelscope.cn/models/damo/cv_nafnet_image-deblur_reds/summary) |


## 期望模型使用方式以及适用范围
本模型适用于智能手机拍摄的带噪图片。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/noisy-demo-0.png'
image_denoise_pipeline = pipeline(Tasks.image_denoising, 'damo/cv_nafnet_image-denoise_sidd')
result = image_denoise_pipeline(img)[OutputKeys.OUTPUT_IMG]
cv2.imwrite('result.png', result)
```

### 模型局限性以及可能的偏差
由于训练数据为SIDD，所有目前的去噪模型对手机拍摄的带噪图片效果良好，而其他类型的噪声可能表现不佳。
## 训练数据介绍
SIDD(Smartphone Image Denoising)-Medium Dataset 包含320个图像对（噪声图像与真值图像），两组图像对来自于同一个场景实例。

Noisy sRGB image (.PNG).

-   Gamma corrected, without any tone mapping.

Ground truth sRGB image (.PNG).
-   Gamma corrected, without any tone mapping.

## 验证数据介绍
SIDD_validation sRGB Images

文件类型：.PNG

文件数量：1280

内容：每32个连续图像块来自同一幅图像，例如图像块0--32来自同一张图像。

SIDD+ Validation sRGB Images

文件类型：.PNG

文件数量：1024

内容：每32个连续图像块来自同一幅图像，例如图像块0--32来自同一张图像。

注意：图像块是标准RGB（sRGB），其已经过gamma校正和全局色调映射。


## 模型训练流程

### 预处理
数据集源地址：

SIDD训练数据集（http://www.cs.yorku.ca/~kamel/sidd/dataset.php)

SIDD-Medium Dataset sRGB images only（~12G）

SIDD验证数据集（https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php）

SIDD+验证数据集（https://competitions.codalab.org/competitions/22231）

推荐使用modelscope dataset托管的SIDD数据集加速下载（https://modelscope.cn/datasets/huizheng/SIDD/summary）
### 训练
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.sidd_image_denoising import \
    SiddImageDenoisingDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_nafnet_image-denoise_sidd'
cache_path = snapshot_download(model_id)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))

# 修改配置文件
def cfg_modify_fn(cfg):
    cfg.train.dataloader.batch_size_per_gpu = 2  # batch size
    cfg.train.dataloader.workers_per_gpu = 4
    
    cfg.train.max_epochs = 1
    cfg.train.optimizer.lr = 1e-5
    return cfg

# 加载数据集
dataset_train = MsDataset.load(
    'SIDD',
    namespace='huizheng',
    subset_name='default',
    split='validation',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
dataset_val = MsDataset.load(
    'SIDD',
    namespace='huizheng',
    subset_name='default',
    split='test',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

dataset_train = SiddImageDenoisingDataset(
            dataset_train, config.dataset, is_train=True)
dataset_val = SiddImageDenoisingDataset(
    dataset_val, config.dataset, is_train=False)

kwargs = dict(
    model=model_id,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    work_dir=tmp_dir,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(default_args=kwargs)
trainer.train()
```

## 使用finetune后模型评估
```python
import os
import tempfile
import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.sidd_image_denoising import \
    SiddImageDenoisingDataset

# 工作目录中包含model card中的所有文件（包括配置文件、模型文件）
tmp_dir = './image_denoise/output/'  # 如果训练过程中的工作目录为`./image_denoise/`，那么验证时的工作目录需要改为 './image_denoise/output/'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_nafnet_image-denoise_sidd'
cache_path = snapshot_download(model_id)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))

# 因为cv_nafnet_image-denoise_sidd模型使用modelscope的trainer类进行训练，其自动保存的模型中每一层的关键字都会有一个`model.`的前缀，
# 我们需要在这里去掉这个前缀。
# 修改 tmp_dir/epoch_1.pth 或 tmp_dir/output/pytorch_model.pt 为模型能够直接加载的文件
def modify_checkpoint(ckpt_path, saved_path):
    input_ckpt = torch.load(ckpt_path)
    
    pretrained_dict = {k.replace('model.', ''): v for k, v in input_ckpt.items() if k.startswith('model.')}
    torch.save(pretrained_dict, saved_path)
    print('successly convert checkpoint!')

modify_checkpoint(tmp_dir+'pytorch_model.pt', tmp_dir+'pytorch_model.pt')

dataset_val = MsDataset.load(
    'SIDD',
    namespace='huizheng',
    subset_name='default',
    split='test',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
eval_dataset = SiddImageDenoisingDataset(
            dataset_val, config.dataset, is_train=False)
kwargs = dict(
    model=tmp_dir,  # 可指定为模型所谓目录，例如 tmp_dir
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()
print(metric_values)
```

## 数据评估及结果
| name | Dataset | PSNR | SSIM |
|:---- |:----    |:---- |:----|
|NAFNet-SIDD-width32|SIDD_val|39.9672|0.9599|
|NAFNet-SIDD-width32|SIDD+_val|36.0885|0.9078|

```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.sidd_image_denoising import \
    SiddImageDenoisingDataset

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_nafnet_image-denoise_sidd'
cache_path = snapshot_download(model_id)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_val = MsDataset.load(
    'SIDD',
    namespace='huizheng',
    subset_name='default',
    split='test',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
eval_dataset = SiddImageDenoisingDataset(
            dataset_val, config.dataset, is_train=False)
kwargs = dict(
    model=model_id,  # 可指定为模型所谓目录，例如 cache_path
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()
print(metric_values)

```

### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@inproceedings{nafnet,
    title = {Simple Baselines for Image Restoration},
    author = {Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
    booktitle = {Proceedings of European Conference on Computer Vision (ECCV)},
    year = {2022}
}
```