
# SCRFD 模型介绍
人脸检测及关键点模型SCRFD

## 模型描述

SCRFD为当前SOTA的人脸检测方法，该方法的主要贡献是从两处入手提升检测器在效率和精度的平衡，分别是： 
- Sample Redistribution(SR), 统计训练数据的人脸size分布，在固定分辨率输入下增广更多小样本来训练shallow stage; 
- Computation Redistribution(CR), 简化搜索空间，采用RegNet的思路对backbone，neck, head网络结构进行搜索; 

通过上述SR和CR两方面，SCRFD family平衡效率和精度，在各算力下均取得SOTA效果，以WIDERFace的hard组为例，在VGA分辨率下，SCRFD-34GF模型的mAP超过竞争方法TinaFace 3.86%，同时GPU推理速度快3倍。SCRFD已被ICLR-2022接收([论文地址](https://arxiv.org/abs/2105.04714), [代码地址](https://github.com/deepinsight/insightface/tree/master/detection/scrfd))。

SCRFD famlity在WIDERFace-Hard的指标如下：

![模型结构](https://modelscope.cn/api/v1/models/damo/cv_resnet_facedetection_scrfd10gkps/repo?Revision=master&FilePath=description/SCRFD-sota.jpg&View=true)

SCRFD方法强化了小目标的检测，但对超大脸(如，部分超出画面)会产生漏检或关键点不准，同时对旋转人脸的效果也未做优化。因此我们使用更大算力的网络，有针对性的优化上述缺点，训练了V2模型(34g_gnkps_v2),相比原模型(10g_bnkps)有了显著提升，使模型可覆盖更多的应用场景。

![对比](https://modelscope.cn/api/v1/models/damo/cv_resnet_facedetection_scrfd10gkps/repo?Revision=master&FilePath=description/compare.jpg&View=true)


## 使用方式和范围

使用方式：
- 推理：输入图片，如存在人脸则返回人脸位置及关键点，可识别多张人脸
- 调优：采用自有数据对模型进行效果优化


目标场景:
- 人脸相关的基础能力，可应用于视频监控/人像美颜/互动娱乐/人脸比对等场景

### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks

face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_resnet_facedetection_scrfd10gkps')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_detection2.jpeg'
result = face_detection(img_path)

# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage
img = LoadImage.convert_to_ndarray(img_path)
cv2.imwrite('srcImg.jpg', img)
img_draw = draw_face_detection_result('srcImg.jpg', result)
import matplotlib.pyplot as plt
plt.imshow(img_draw)
```

### 数据集
- WIDERFACE: 本模型采用开源数据集WIDERFACE训练并已上传至ModelScope的DatasetHub, 详情请见[WIDER_FACE](https://modelscope.cn/datasets/shaoxuan/WIDER_FACE),推荐使用；
- 自有数据:请按照如下格式准备自有数据的标注信息
```
# <image_path> image_width image_height
bbox_x1 bbox_y1 bbox_x2 bbox_y2 (<keypoint,3>*N)
...
...
# <image_path> image_width image_height
bbox_x1 bbox_y1 bbox_x2 bbox_y2 (<keypoint,3>*N)
...
...
```

### 模型训练
通过使用托管在modelscope DatasetHub上的数据集WIDER_FACE进行训练：
```python
import os
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.hub.snapshot_download import snapshot_download

model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'
ms_ds_widerface = MsDataset.load('WIDER_FACE_mini', namespace='shaoxuan')  # remove '_mini' for full dataset

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
        cfg.data.samples_per_gpu = 4
        return cfg

kwargs = dict(
        cfg_file=os.path.join(cache_path, 'mmcv_scrfd.py'),
        work_dir=tmp_dir,
        train_root=train_root,
        val_root=val_root,
        total_epochs=1,  # run #epochs
        cfg_modify_fn=_cfg_modify_fn)

trainer = build_trainer(name=Trainers.face_detection_scrfd, default_args=kwargs)
trainer.train()
```
- 更多示例(如，多卡训练)请参阅：`tests/trainers/test_face_detection_scrfd_trainer.py`
- 本模型使用8卡v100，使用SGD优化器，lr=0.02，在440/544epoch时降低10倍学习率，并在640epoch时产出模型，数据增强方面相比原论文，增加了0.2概率的极大脸采样，以及0.3概率的图像旋转，其余超参与论文一致，详见`mmcv_scrfd.py`

## 模型性能指标

模型在WIDERFaces数据集(VGA分辨率输入)的评测指标、模型大小、推理耗时(2080ti)如下:

| Name | Easy | Medium | Hard | FLOPS | Params(M) | Infer(ms) |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
| SCRFD_10G_BNKPS | 95.40 | 94.01 | 82.80 | 10G | 4.23 | 5.0|
| SCRFD_34G_GNKPS_v2 | 96.17 | 95.19 | 84.88 | 34G | 9.84 | 11.8|

## 来源说明
本模型及代码来自开源社区([地址](https://github.com/deepinsight/insightface/tree/master/detection/scrfd))，请遵守相关许可。

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
