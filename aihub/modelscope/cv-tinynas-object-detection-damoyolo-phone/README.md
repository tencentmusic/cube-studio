
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# 实时手机检测-通用 模型介绍
<div align="center"><img src="assets/results.png" width="2000"></div>

本模型为**高性能热门应用系列检测模型**中的 <u>实时手机检测模型</u>，基于面向工业落地的高性能检测框架[DAMOYOLO](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary)，其精度和速度超越当前经典的YOLO系列方法。用户使用的时候，仅需要输入一张图像，便可以获得图像中所有手机的坐标信息，并可用于打电话检测等后续应用场景。


## 相关模型

以下是ModelScope上所有的热门应用检测模型（垂类目标检测模型）：

| 序号 | 模型名称 | 序号 | 模型名称 |
| ------------ | ------------ | ------------ | ------------ |
| 1 | [实时人体检测模型](https://modelscope.cn/models/damo/cv_tinynas_human-detection_damoyolo/summary) | 6 | [实时香烟检测模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_cigarette/summary) |
| 2 | [实时人头检测模型](https://modelscope.cn/models/damo/cv_tinynas_head-detection_damoyolo/summary) | 7 | [实时手机检测模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_phone/summary) |
| 3 | [实时手部检测模型](https://modelscope.cn/models/damo/cv_yolox-pai_hand-detection/summary) | 8 | [实时交通标识检测模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_traffic_sign/summary) |
| 4 | [实时口罩检测模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_facemask/summary) | 9 | Coming soon |
| 5 | [实时安全帽检测模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_safety-helmet/summary) |


## 模型描述
本模型为实时手机检测模型，基于检测框架[DAMOYOLO-S模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary)，DAMO-YOLO是一个面向工业落地的目标检测框架，兼顾模型速度与精度，其训练的模型效果超越了目前的一众YOLO系列方法，并且仍然保持极高的推理速度。DAMOYOLO与YOLO系列其它经典工作的性能对比如下图所示：

<div align="center"><img src="https://modelscope.cn/api/v1/models/damo/cv_tinynas_object-detection_damoyolo_phone/repo?Revision=master&FilePath=assets/DAMOYOLO_performance.jpg&View=true" width="500"></div>

DAMOYOLO整体网络结构如下，整体由backbone (MAE-NAS), neck (GFPN), head (ZeroHead)三部分组成，基于"large neck, small head"的设计思想，对低层空间信息和高层语义信息进行更加充分的融合，从而提升模型最终的检测效果。

<div align="center"><img src="https://modelscope.cn/api/v1/models/damo/cv_tinynas_object-detection_damoyolo_phone/repo?Revision=master&FilePath=assets/DAMOYOLO_architecture.jpg&View=true" width="2000"></div>


## 期望模型使用方式以及适用范围
该模型适用于手机检测，输入任意的图像，输出图像中手机的位置信息。

| 类别ID | 类别名称 |
| ------------ | ------------ |
| 1 | phone |

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型，得到图像中所有手机的外接矩形框信息。

#### 推理代码范例
基础示例代码。下面的示例代码展示的是如何通过一张图片作为输入，得到图片对应的手机检测结果。
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_tinynas_object-detection_damoyolo_phone'
input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_phone.jpg'

phone_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)
result = phone_detection(input_location)
print("result is : ", result)
```

#### 微调代码范例
下面的示例代码展示如何基于已有的COCO格式数据集，进行模型的微调（finetune）。详细的训练说明可以参见：[DAMOYOLO-S模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary)。
```python
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

# Step 1: 数据集准备，可以使用modelscope上已有的数据集，也可以自己在本地构建COCO数据集
train_dataset = MsDataset.load('phone_detection_for_train', namespace="modelscope", split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
val_dataset = MsDataset.load('phone_detection_for_train', namespace="modelscope", split='validation', download_mode=DownloadMode.FORCE_REDOWNLOAD)

# Step 2: 相关参数设置
train_root_dir = train_dataset.config_kwargs['split_config']['train']
val_root_dir = val_dataset.config_kwargs['split_config']['validation']
train_img_dir = osp.join(train_root_dir, 'images')
val_img_dir = osp.join(val_root_dir, 'images')
train_anno_path = osp.join(train_root_dir, 'train.json')
val_anno_path = osp.join(val_root_dir, 'val.json')
kwargs = dict(
            model='damo/cv_tinynas_object-detection_damoyolo_phone', # 使用DAMO-YOLO-S模型 
            gpu_ids=[  # 指定训练使用的gpu
                0,
            ],
            batch_size=2, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=3, # 总的训练epochs
            num_classes=1, # 自定义数据中的类别数
            load_pretrain=True, # 是否载入预训练模型，若为False，则为从头重新训练
            base_lr_per_img=0.001, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=train_img_dir, # 训练图片路径
            val_image_dir=val_img_dir, # 测试图片路径
            train_ann=train_anno_path, # 训练标注文件路径
            val_ann=val_anno_path, # 测试标注文件路径
            )

# Step 3: 开启训练任务
trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
trainer.train()
```

## 训练数据介绍
本模型是基于以下数据集训练得到：

- phone-internal: 内部积累以及互联网爬取的相关手机数据。

## 数据评估及结果
模型在phone-internal的验证集上客观指标如下：

| Method | AP@0.5 | Latency(ms)<br>T4-TRT-FP16| FLOPs (G)| Parameters (M)|
| ------------ | ------------ | ------------ | ------------ | ------------ |
| **DAMO-YOLO-S** | 0.888 | 3.83 | 37.8 | 16.3  |

### 相关论文以及引用信息
本模型主要参考论文如下（论文链接：[link](https://arxiv.org/abs/2211.15444)）：

```BibTeX
 @article{damoyolo,
  title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
  author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
  journal={arXiv preprint arXiv:2211.15444v2},
  year={2022}
}
```
