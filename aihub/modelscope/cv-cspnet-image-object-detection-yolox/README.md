
# 通用实时检测模型

通用实时检测为目标检测的子任务，本模型为高性能通用实时检测模型，提供快速、精确的目标检测能力。

See the world fast and accurately!




## 模型描述
<img src="res/demo.png" >


YOLOX为YOLO检测系列的最近增强版本。在实时通用检测模型中，YOLO系列模型获得显著的进步，大大地推动了社区的发展。YOLOX在原有YOLO系列的基础上，结合无锚点（anchor-free）设计，自动优化GT分配（SimOTA）策略，分类回归头解耦（Decoupling Head）等一系列前沿视觉检测技术，显著地提高了检测模型的准确度与泛化能力，将当前的目标检测水平推到了一个新的高度。本模型为YOLOX的小规模模型，基于公开数据集COCO训练，支持80类通用目标检测。


<img src="res/git_fig.png" width="1000" >

## 期望模型使用方式以及适用范围

- 日常通用检测场景目标定位与识别。
- 日常通用场景目标粗略计数。
- 作为其他日常场景算法的前置算法，如人体关键点检测，行为识别等。

### 如何使用


在ModelScope框架上，可以通过ModelScope的pipeline进行调用。

Now, you can play the model with a few line codes!

#### 推理代码范例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
基础示例代码。下面的示例代码展示的是如何通过一张图片作为输入，得到图片对应的检测结果坐标信息。
```python
# 模型推理
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

realtime_detector = pipeline(Tasks.image_object_detection, model='damo/cv_cspnet_image-object-detection_yolox')
result = realtime_detector('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg')
# 打印结果
print(result)

### 可视化代码
# def vis_det_img(input_path, res):
#     def get_color(idx):
#         idx = (idx + 1) * 3
#         color = ((10 * idx) % 255, (20 * idx) % 255, (30 * idx) % 255)
#         return color
#     img = cv2.imread(input_path)
#     unique_label = list(set(res['labels']))
#     for idx in range(len(res['scores'])):
#         x1, y1, x2, y2 = res['boxes'][idx]
#         score = str(res['scores'][idx])
#         label = str(res['labels'][idx])
#         color = get_color(unique_label.index(label))
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#         cv2.putText(img, label, (int(x1), int(y1) - 10),
#                     cv2.FONT_HERSHEY_PLAIN, 1, color)
#         cv2.putText(img, score, (int(x1), int(y2) + 10),
#                     cv2.FONT_HERSHEY_PLAIN, 1, color)
#     return img

# im_vis = vis_det_img('./image_detection.jpg', result)
# cv2.imwrite('./image_detection_vis.jpg', im_vis)
```

#### 微调代码范例
下面的示例代码展示如何基于已有的COCO格式数据集，进行模型的微调（finetune）。
```python
# 模块引入
import os
import cv2
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.hub.snapshot_download import snapshot_download

# 数据集准备
train_dataset = MsDataset.load(
    'head_detection_for_train_tutorial', 
    namespace="modelscope", 
    split='train', 
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
val_dataset = MsDataset.load(
    'head_detection_for_train_tutorial', 
    namespace="modelscope", 
    split='validation', 
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

# 相关参数设置
model_id = 'damo/cv_cspnet_image-object-detection_yolox'
work_dir = "./output/cv_cspnet_image-object-detection_yolox_1"
trainer_name = Trainers.easycv
batchsize = 8
total_epochs = 15
class_names = ['head']

cfg_options = {
    'train.max_epochs':
    total_epochs,
    'train.dataloader.batch_size_per_gpu':
    batchsize,
    'evaluation.dataloader.batch_size_per_gpu':
    1,
    'train.optimizer.lr': 0.01 / 64 * batchsize,
    'train.lr_scheduler.warmup_iters': 1,
    'train.lr_scheduler.num_last_epochs': 5,
    'train.hooks': [
        {
            'type': 'CheckpointHook',
            'interval': 5
        },
        {
            'type': 'EvaluationHook',
            'interval': 5
        },
        {
            'type': 'TextLoggerHook',
            'ignore_rounding_keys': None,
            'interval': 1
        },
    ],
    'dataset.train.data_source.classes': class_names,
    'dataset.val.data_source.classes': class_names,
    'evaluation.metrics.evaluators': [
        {
            'type': 'CocoDetectionEvaluator',
            'classes': class_names
        }
    ],
    'CLASSES': class_names,
}

# 新建trainer
kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    work_dir=work_dir,
    cfg_options=cfg_options)
print("build trainer.")
trainer = build_trainer(trainer_name, kwargs)
print("start training.")

# 预训练模型下载
cache_path = snapshot_download(model_id)

# 开启训练
trainer.train(
    checkpoint_path=os.path.join(cache_path, 'pytorch_model.pt'),
    load_all_state=False,
)


# 模型评估
eval_res = trainer.evaluate(checkpoint_path=os.path.join(work_dir, 'epoch_15.pth'))
print(eval_res)
```

### 模型局限性以及可能的偏差
  - 对于小物体的检测在某些困难场景会存在效果差的情况，建议根据场景对检出小目标进行分数过滤。
  - 目前模型仅限于pipeline调用，尚未支持finetune和evaluation。
  - 复杂专用场景性能会产生显著下降，如复杂视角、超低光照以及严重遮挡等。
  - 当前版本在Python 3.7环境测试通过，其他环境下可用性待测试。

## 训练数据介绍
<img src="res/coco-logo.png" width="1000">

<img src="res/coco-examples.jpeg" width="1000">

本模型基于COCO数据集的目标检测部分数据及标注进行训练。COCO数据集的全称是[Microsoft Common Objects in Context](https://cocodataset.org/#home)， 是一个评估计算机视觉模型性能的“黄金”标准基准数据集，旨在推动目标检测、实例分割、看图说话和人物关键点方面的研究。其中目标检测任务有90个日常常见类别，在学术研究中常用其中的80类作为基准的评测数据集。

## 模型训练流程

模型在线训练暂不支持。部分关键训练细节如下：
- 使用 SGD 优化算法，cos LR scheduler，warmup策略。
- 训练迭代为 300个 epoch，其中最后15个epoch关闭数据增强。
- Mosaic，颜色增强等策略被应用到训练预处理中。

## 输入预处理

- 输入图像根据长边resize到640后，padding 为640x640的矩形进行推理。
- 图像归一化。

## 数据评估及结果
|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX](https://arxiv.org/pdf/2107.08430.pdf)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [Official](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) 


## 引用

如您的相关著作、作品使用了该模型，请引用以下信息：

```
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

