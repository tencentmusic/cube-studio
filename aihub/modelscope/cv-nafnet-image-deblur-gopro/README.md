
# NAFNet: Nonlinear Activation Free Network for Image Restoration

## 模型描述
NAFNet（Nonlinear Activation Free Network）提出了一个简单的基线，计算效率高。其不需要使用非线性激活函数（Sigmoid、ReLU、GELU、Softmax等），可以达到SOTA性能。其网络结构如下图所示：

<img src="./data/nafnet_arch.png" width=224 alt="NAFNet architecture">


| <img src="./data/deblur.gif"  width=512 alt="NAFNet For Image Deblur">|
| :-----------------------------------------------------------------------------------: |
|                                     Deblur                                           |

## 期望模型使用方式以及适用范围
本模型适用于运动模糊图像。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/GOPR0384_11_00-000001.png'
image_deblur_pipeline = pipeline(Tasks.image_deblurring, 'damo/cv_nafnet_image-deblur_gopro')
result = image_deblur_pipeline(img)[OutputKeys.OUTPUT_IMG]
cv2.imwrite('result.png', result)
```

### 模型局限性以及可能的偏差
由于训练数据为GOPRO，所有目前的去模糊模型对具有运动模糊图片效果良好，而其他类型的模糊可能表现不佳。
## 训练数据介绍


## 验证数据介绍



## 模型训练流程

### 预处理
数据集源地址：

GOPRO_Large数据集原始地址（https://seungjunnah.github.io/Datasets/gopro)

推荐使用modelscope dataset托管的GOPRO数据集加速下载（https://modelscope.cn/datasets/damo/GOPRO/summary）
### 训练
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.gopro_image_deblurring_dataset import \
    GoproImageDeblurringDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_nafnet_image-deblur_gopro'
cache_path = snapshot_download(model_id)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_train = MsDataset.load(
    'GOPRO',
    namespace='damo',
    subset_name='default',
    split='test',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
dataset_val = MsDataset.load(
    'GOPRO',
    namespace='damo',
    subset_name='subset',
    split='test',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

dataset_train = GoproImageDeblurringDataset(
            dataset_train, config.dataset, is_train=True)
dataset_val = GoproImageDeblurringDataset(
    dataset_val, config.dataset, is_train=False)

kwargs = dict(
    model=model_id,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
trainer.train()
```

## 数据评估及结果
| name | Dataset | PSNR | SSIM |
|:---- |:----    |:---- |:----|
|NAFNet-GoPro-width64|GoPro_test|33.7103|0.9668|

```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.gopro_image_deblurring_dataset import \
    GoproImageDeblurringDataset

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_nafnet_image-deblur_gopro'
cache_path = snapshot_download(model_id)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))
dataset_val = MsDataset.load(
    'GOPRO',
    namespace='damo',
    subset_name='subset',
    split='test',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
eval_dataset = GoproImageDeblurringDataset(
            dataset_val, config.dataset, is_train=False)
kwargs = dict(
    model=model_id,
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
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_nafnet_image-deblur_gopro.git
```