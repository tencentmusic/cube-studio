

# 读光文字识别
## News
- 2023年3月：
    - 新增训练/微调流程，支持自定义参数及数据集，详见代码示例。
- 2023年2月：
    - 新增业界主流[CRNN-通用场景](https://www.modelscope.cn/models/damo/cv_crnn_ocr-recognition-general_damo/summary)模型。

## 传送门
各场景文本识别模型：
- [ConvNextViT-通用场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-general_damo/summary)
- [ConvNextViT-手写场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/summary)
- [ConvNextViT-自然场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-scene_damo/summary)
- [ConvNextViT-文档印刷场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-document_damo/summary)
- [CRNN-通用场景](https://www.modelscope.cn/models/damo/cv_crnn_ocr-recognition-general_damo/summary)

各场景文本检测模型：
- [SegLink++-通用场景行检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-line-level_damo/summary)
- [SegLink++-通用场景单词检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-word-level_damo/summary)
- [DBNet-通用场景行检测](https://www.modelscope.cn/models/damo/cv_resnet18_ocr-detection-db-line-level_damo/summary)

整图OCR能力：
- [整图OCR-多场景](https://modelscope.cn/studios/damo/cv_ocr-text-spotting/summary)

欢迎使用！

## 模型描述
- 文字识别，即给定一张文本图片，识别出图中所含文字并输出对应字符串。
- 本模型主要包括三个主要部分，Convolutional Backbone提取图像视觉特征，ConvTransformer Blocks用于对视觉特征进行上下文建模，最后连接CTC loss进行识别解码以及网络梯度优化。识别模型结构如下图：   

<p align="center">
    <img src="https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo/repo?Revision=master&FilePath=./resources/ConvTransformer-Pipeline.jpg&View=true"/> 
</p>

## 期望模型使用方式以及适用范围
本模型主要用于给输入图片输出图中文字内容，具体地，模型输出内容以字符串形式输出。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。
- 注：输入图片应为包含文字的单行文本图片。其它如多行文本图片、非文本图片等可能没有返回结果，此时表示模型的识别结果为空。

## 模型推理
在安装完成ModelScope之后即可使用ocr-recognition的能力。(在notebook的CPU环境或GPU环境均可使用)
- 使用图像的url，或准备图像文件上传至notebook（可拖拽）。
- 输入下列代码。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo')

### 使用url
img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_licenseplate//ocr_recognition_licenseplate.jpg'
result = ocr_recognition(img_url)
print(result)

### 使用图像文件
### 请准备好名为'ocr_recognition_licenseplate.jpg'的图像文件
# img_path = 'ocr_recognition_licenseplate.jpg'
# img = cv2.imread(img_path)
# result = ocr_recognition(img)
# print(result)
```

### 模型可视化效果
以下为模型的可视化文字识别效果。

<p align="center">
    <img src="https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo/repo?Revision=master&FilePath=./resources/rec_result_visu.jpg&View=true" width="300" /> 
</p>

### 模型局限性以及可能的偏差
- 模型是在中英文数据集上训练的，在其他语言的数据上有可能产生一定偏差，请用户自行评测后决定如何使用。
- 当前版本在python3.7的CPU环境和单GPU环境测试通过，其他环境下可用性待测试。

## 模型微调/训练
### 训练数据及流程介绍
- 本文字识别模型训练数据集是收集数据以及合成数据，训练数据数量约1M。
- 本模型参数随机初始化，然后在训练数据集上进行训练，在32x300尺度下训练20个epoch。

### 模型微调/训练示例
#### 训练数据集准备
示例采用[ICDAR13手写数据集](https://modelscope.cn/datasets/damo/ICDAR13_HCTR_Dataset/summary)，已制作成lmdb，数据格式如下
```
'num-samples': number,
'image-000000001': imagedata,
'label-000000001': string,
...
```
详情可下载解析了解。

#### 配置训练参数并进行微调/训练
参考代码及详细说明如下
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile, DownloadMode

### 请确认您当前的modelscope版本，训练/微调流程在modelscope==1.4.0及以上版本中 
### 当前notebook中版本为1.3.2，请手动更新，建议使用GPU环境

model_id = 'damo/cv_convnextTiny_ocr-recognition-licenseplate_damo' 
cache_path = snapshot_download(model_id) # 模型下载保存目录
config_path = os.path.join(cache_path, ModelFile.CONFIGURATION) # 模型参数配置文件，支持自定义
cfg = Config.from_file(config_path)

# 构建数据集，支持自定义
train_data_cfg = ConfigDict(
    name='ICDAR13_HCTR_Dataset', 
    split='test',
    namespace='damo',
    test_mode=False)

train_dataset = MsDataset.load( 
    dataset_name=train_data_cfg.name,
    split=train_data_cfg.split,
    namespace=train_data_cfg.namespace,
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

test_data_cfg = ConfigDict(
    name='ICDAR13_HCTR_Dataset',
    split='test',
    namespace='damo',
    test_mode=True)

test_dataset = MsDataset.load(
    dataset_name=test_data_cfg.name,
    split=test_data_cfg.split,
    namespace=train_data_cfg.namespace,
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

tmp_dir = tempfile.TemporaryDirectory().name # 模型文件和log保存位置，默认为"work_dir/"

# 自定义参数，例如这里将max_epochs设置为15，所有参数请参考configuration.json
def _cfg_modify_fn(cfg):
    cfg.train.max_epochs = 15
    return cfg

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    work_dir=tmp_dir,
    cfg_modify_fn=_cfg_modify_fn)

# 模型训练
trainer = build_trainer(name=Trainers.ocr_recognition, default_args=kwargs)
trainer.train()
```