

# 读光文字检测
## News
- 2023年3月：
    - 新增DBNet训练/微调流程，支持自定义参数及数据集，详见代码示例。
- 2023年2月：
    - 新增业界主流[DBNet-通用场景](https://www.modelscope.cn/models/damo/cv_resnet18_ocr-detection-db-line-level_damo/summary)模型。

## 传送门
各场景文本检测模型：
- [SegLink++-通用场景行检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-line-level_damo/summary)
- [SegLink++-通用场景单词检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-word-level_damo/summary)
- [DBNet-通用场景行检测](https://www.modelscope.cn/models/damo/cv_resnet18_ocr-detection-db-line-level_damo/summary)

各场景文本识别模型：
- [ConvNextViT-手写场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/summary)
- [ConvNextViT-手写场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/summary)
- [ConvNextViT-文档印刷场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-document_damo/summary)
- [ConvNextViT-自然场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-scene_damo/summary)
- [ConvNextViT-车牌场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo/summary)
- [CRNN-通用场景](https://www.modelscope.cn/models/damo/cv_crnn_ocr-recognition-general_damo/summary)

整图OCR能力：
- [整图OCR-多场景](https://modelscope.cn/studios/damo/cv_ocr-text-spotting/summary)

欢迎使用！

## 模型描述

本模型是基于分割的文字检测方法，把文字行的区域分割文字中心区域和文字边界区域，通过处理得到文字完整区域，最后得到文字区域的外接框。详见：[DBNet(Paper)](https://arxiv.org/pdf/1911.08947.pdf)


## 期望模型使用方式以及适用范围
本模型主要用于给输入图片输出图中文字外接框坐标，具体地，模型输出的框的坐标为文字框四边形的四个角点的坐标，左上角为第一个点，按照顺时针的顺序依次输出各个点的坐标，分别为(x1,y1)(x2,y2)(x3,y3)(x4,y4)。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope之后即可使用ocr-detection的能力。

### 预处理和后处理
测试时的主要预处理和后处理如下：
- Resize Pad（预处理）: 输入图片短边resize到736，短边等比例缩放，并且补pad到长短边相等
- threshold后处理）: thresh和box_thresh采用0.2和0.3值

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-db-line-level_damo')
result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg')
print(result)
```

### 完整OCR能力体验
如果想体验完整的OCR能力，对整图中的文字进行检测识别，可以体验[创空间应用](https://modelscope.cn/studios/damo/cv_ocr-text-spotting/summary)。对于文字检测模型和文字识别模型的串联，可以参考[说明文档](https://modelscope.cn/dynamic/article/42)。

### 模型局限性以及可能的偏差
- 模型是在特定中英文数据集上训练的，在其他场景和语言的数据上有可能产生一定偏差，请用户自行评测后决定如何使用。
- 当前版本在python3.7环境CPU和单GPU环境测试通过，其他环境下可用性待测试

## 模型训练

### 训练数据和训练流程简介
本模型行检测模型训练数据集是MTWI/ReCTS/SROIE/LSVT，训练数据数量约53K。本模型利用imagenet预训练参数进行初始化，然后在训练数据集上进行训练，先利用640x640尺度训练200epoch。

### 模型微调训练示例
支持使用自定义数据对DBNet行检测模型进行微调训练。

#### 训练数据准备
准备训练数据和测试数据(比如发票数据集[SROIE](https://rrc.cvc.uab.es/?ch=13&com=tasks))，数据目录结构如下
```
├── custom_data
│   ├── train_list.txt
│   ├── train_images
│   │   └── 1.jpg
│   ├── train_gts
│   │   └── 1.txt
│   ├── test_list.txt
│   ├── test_images
│   │   └── 2.jpg
│   ├── test_gts
│   │   └── 2.txt
```
其中，train_list.txt（以及test_list.txt）每行为图片文件名，如下所示：
```
1.jpg
```
标注格式采用ICDAR2015的格式，即标注文件1.txt每行为‘文字框坐标+识别标签’的格式，如下所示：
```
482.0,524.0,529.0,524.0,529.0,545.0,482.0,545.0,8.70
556.0,525.0,585.0,525.0,585.0,546.0,556.0,546.0,SR
```

#### 设定训练参数配置，进行微调训练
设定相关配置参数，运行代码进行微调训练，训练结果保存在'./workdirs'目录下。

```python
### 请确认您当前的modelscope版本，训练/微调流程在modelscope==1.4.0及以上版本中 
### 当前notebook中版本为1.3.2，请手动更新，建议使用GPU环境
import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.hub.snapshot_download import snapshot_download

model_id = 'damo/cv_resnet18_ocr-detection-db-line-level_damo'
cache_path = snapshot_download(model_id) # 模型下载保存目录
config_file = os.path.join(cache_path, 'configuration.json') # 模型参数配置文件，可以自定义
pretrained_model = os.path.join(cache_path, 'db_resnet18_public_line_640x640.pt') # 预训练模型
saved_dir = './workdirs' # 训练结果保存目录
saved_finetune_model = os.path.join(saved_dir, 'final.pt') # 训练保存的模型路径
saved_infer_model = os.path.join(saved_dir, 'pytorch_model.pt') # 训练模型转换成推理模型的路径

kwargs = dict(
    cfg_file=config_file,
    gpu_ids=[
        0,
    ],
    batch_size=8,
    max_epochs=5,
    base_lr=0.007,
    load_pretrain=True,
    pretrain_model=pretrained_model,
    cache_path=cache_path,
    train_data_dir=['./custom_data/'],
    train_data_list=[
        './custom_data/train_list.txt'
    ],
    val_data_dir=['./custom_data/'],
    val_data_list=['./custom_data/test_list.txt'])
trainer = build_trainer(
    name=Trainers.ocr_detection_db, default_args=kwargs)
trainer.train()

```

#### 模型评测和优化模型推理
完成微调训练后，对测试集进行评测，支持ICDAR15评测标准。
```python
# 接上代码
trainer.evaluate(checkpoint_path=saved_finetune_model)
```

也可以使用微调训练后的模型对单张图片进行推理测试。
```python
# 接上代码
cmd = 'cp {} {}'.format(config_file, saved_dir)
os.system(cmd)
ocr_detection = pipeline(Tasks.ocr_detection, model=saved_dir)
result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg')
print(result)
```

## 引用

```latex
 @inproceedings{liao2020real,
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  title={Real-time Scene Text Detection with Differentiable Binarization},
  booktitle={Proc. AAAI},
  year={2020}
}
```