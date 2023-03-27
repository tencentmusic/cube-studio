
# 卡证检测矫正模型介绍


## 模型描述

在实人认证、文档电子化等场景中需要自动化提取卡证的信息，以便进一步做录入处理。这类场景通常存在两类问题，一是识别卡证类型时易受背景干扰，二是卡证拍摄角度造成的文字畸变影响OCR准确率。鉴于证件类数据的敏感性，我们采用大量合成卡证数据做训练(参见：[SyntheticCards](https://modelscope.cn/datasets/shaoxuan/SyntheticCards)), 并改造人脸检测SOTA方法SCRFD([论文地址](https://arxiv.org/abs/2105.04714), [代码地址](https://github.com/deepinsight/insightface/tree/master/detection/scrfd))训练了卡证检测矫正模型，可以对各类国际常见卡证（如，身份证、护照、驾照等）进行检测、定位及矫正，得到去除背景的正视角卡证图像，便于后续卡证分类或OCR内容提取。 

### 训练数据：
![训练数据](https://modelscope.cn/api/v1/models/damo/cv_resnet_carddetection_scrfd34gkps/repo?Revision=master&FilePath=description/traindata.jpg&View=true)

### 效果展示：
![效果展示](https://modelscope.cn/api/v1/models/damo/cv_resnet_carddetection_scrfd34gkps/repo?Revision=master&FilePath=description/card_detect.jpg&View=true)

## 使用方式和范围

使用方式：
- 推理：输入图片，如存在卡证则返回卡证及角点位置，以及每个矫正后的卡证图片
- 调优：采用自有数据对模型进行效果优化


目标场景:
- 卡证相关的前置基础能力，可应用于卡证OCR/证件分类/证件防伪等场景

### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks

card_detection = pipeline(Tasks.card_detection, 'damo/cv_resnet_carddetection_scrfd34gkps')
img_path = 'https://design3d.oss-cn-qingdao.aliyuncs.com/MS_test_img/card_detection.jpg'
result = card_detection(img_path)

# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import draw_card_detection_result
from modelscope.preprocessors.image import LoadImage
import matplotlib.pyplot as plt
img = LoadImage.convert_to_ndarray(img_path)
cv2.imwrite('srcImg.jpg', img)
img_list = draw_card_detection_result('srcImg.jpg', result)
for i, img in enumerate(img_list):
    plt.figure()
    plt.imshow(img_list[i])
```

### 数据集
- SyntheticCards: 采用开源数据素材合成的虚拟卡证数据，并已上传至ModelScope的DatasetHub, 详情请见[SyntheticCards](https://modelscope.cn/datasets/shaoxuan/SyntheticCards)；
- 自有数据:如需使用自己的数据优化模型，请按照如下格式准备标注信息，其中角点顺序为左下、左上、右上、右下，每个角点格式为(x,y,1)
```
# <image_path> image_width image_height
bbox_x1 bbox_y1 bbox_x2 bbox_y2 (<keypoint,3>*4)
...
...
# <image_path> image_width image_height
bbox_x1 bbox_y1 bbox_x2 bbox_y2 (<keypoint,3>*4)
...
...
```

### 模型训练
通过使用托管在modelscope DatasetHub上的数据集SyntheticCards进行训练：
```python
import os
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.hub.snapshot_download import snapshot_download

model_id = 'damo/cv_resnet_carddetection_scrfd34gkps'
ms_ds_widerface = MsDataset.load('SyntheticCards_mini', namespace='shaoxuan')  # remove '_mini' for full dataset

data_path = ms_ds_widerface.config_kwargs['split_config']
train_dir = data_path['train']
val_dir = data_path['validation']

def get_name(dir_name):
    names = [i for i in os.listdir(dir_name) if not i.startswith('_')]
    return names[0]

train_root = train_dir + '/' + get_name(train_dir) + '/'
val_root = val_dir + '/' + get_name(val_dir) + '/'
cache_path = snapshot_download(model_id)
tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    
def _cfg_modify_fn(cfg):
        cfg.checkpoint_config.interval = 1
        cfg.log_config.interval = 10
        cfg.evaluation.interval = 1
        cfg.data.workers_per_gpu = 1
        cfg.data.samples_per_gpu = 2
        return cfg

kwargs = dict(
        cfg_file=os.path.join(cache_path, 'mmcv_scrfd.py'),
        work_dir=tmp_dir,
        train_root=train_root,
        val_root=val_root,
        total_epochs=1,  # run #epochs
        cfg_modify_fn=_cfg_modify_fn)

trainer = build_trainer(name=Trainers.card_detection_scrfd, default_args=kwargs)
trainer.train()
```
- 更多示例(如，多卡训练)请参阅：`tests/trainers/test_card_detection_scrfd_trainer.py`
- 本模型使用8卡v100，使用SGD优化器，lr=0.02，在120/200/240epoch时降低10倍学习率，并在280epoch时产出模型, 其余训练超参数详见`mmcv_scrfd.py`


## 来源说明
模型训练方法基于[SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)，训练数据使用了[SyntheticCards](https://modelscope.cn/datasets/shaoxuan/SyntheticCards)，请遵守相关许可。

## 引用
如果你觉得该模型有所帮助，请考虑引用下面的相关的论文：
```
@article{guo2021sample,
  title={Sample and Computation Redistribution for Efficient Face Detection},
  author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2105.04714},
  year={2021}
}
```
