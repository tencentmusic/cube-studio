
# DamoFD-10G 模型介绍
轻量级人脸检测模型DamoFD-10G。

## 模型描述
Motivation: 目前的Nas方法主要由两个模块组成，网络生成器和精度预测器。其中网络生成器用于生成候选的backbone结构，精度预测器用来对采样的backbone结构预测精度。由于检测和分类的任务目标不一致，前者更重视backbone stage-level (c2-c5)的表征，而后者更重视high-level(c5)的表征，这就导致了用于分类任务上的精度预测器擅长预测high-level的表征能力而无法预测stage-level的表征能力。因此，在人脸检测任务上，我们需要一个可以预测stage-level表征能力的精度预测器来更好的搜索face detection-friendly backbone。DamoFD: 针对如何设计可以预测stage-level表征能力的精度预测器，我们从刻画network expressivity的角度出发，创新性地提出了SAR-score来无偏的刻画stage-wise network expressivity，同时基于数据集gt的先验分布，来确定不同stage的重要性，进一步提出了DDSAR-score 来刻画detection backbone的精度。论文已被ICLR2023接收，更多代码细节以及modelscope应用可以前往[EasyFace_DamoFD](https://github.com/ly19965/FaceMaas/tree/master/face_project/face_detection/DamoFD)。

## 模型使用方式和使用范围
本模型可以检测输入图片中人脸的位置。

### 安装
```
conda create -n EasyFace python=3.7
conda activate EasyFace
# pytorch >= 1.3.0
pip install torch==1.8.1+cu102  torchvision==0.9.1+cu102 torchaudio==0.8.1  --extra-index-url https://download.pytorch.org/whl/cu102
git clone https://github.com/ly19965/FaceMaas
cd FaceMaas
pip install -r requirements/tests.txt 
pip install -r requirements/framework.txt
pip install -r requirements/cv.txt 
```
### 代码范例
#### 推理
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks

face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd-10G')
# 支持 url image and abs dir image path
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_detection2.jpeg' 
result = face_detection(img_path)

# 提供可视化结果
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage
img = LoadImage.convert_to_ndarray(img_path)
cv2.imwrite('srcImg.jpg', img)
img_draw = draw_face_detection_result('srcImg.jpg', result)
import matplotlib.pyplot as plt
plt.imshow(img_draw)
```

#### 训练
```python
import os
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.hub.snapshot_download import snapshot_download

model_id = 'damo/cv_ddsar_face-detection_iclr23-damofd-10G'
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
    cfg_file=os.path.join(cache_path, 'DamoFD_lms.py'),
    work_dir=tmp_dir,
    train_root=train_root,
    val_root=val_root,
    total_epochs=1,  # run #epochs
    cfg_modify_fn=_cfg_modify_fn)

trainer = build_trainer(name=Trainers.face_detection_scrfd, default_args=kwargs)
trainer.train()
```


### 目标场景
- 人脸相关的基础能力，可应用于人像美颜/互动娱乐/人脸比对等场景, 尤其适应于端上人脸应用场景


### 模型局限性及可能偏差
- 模型size较小，模型鲁棒性可能有所欠缺。
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 预处理
测试时主要的预处理如下：
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

### 模型训练流程
- 本模型使用8卡v100，使用SGD优化器，lr=0.02，在440/544epoch时降低10倍学习率，并在640epoch时产出模型，详见DamoFD_lms.py

### 测试集
- WIDERFACE： 测试集已上传至ModelScope的DatasetHub，详情请见[WIDER_FACE](https://modelscope.cn/datasets/shaoxuan/WIDER_FACE)。

## 数据评估及结果
模型在WiderFace的验证集上客观指标如下：
| Method | Easy | Medium | Hard |
| ------------ | ------------ | ------------ | ------------ |
| DamoFD-10G | 95.14  | 94.29 | 84.07 |

## 人脸相关模型

以下是ModelScope上人脸相关模型:

- 人脸检测

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [RetinaFace人脸检测模型](https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary) |
| 2 | [MogFace人脸检测模型-large](https://modelscope.cn/models/damo/cv_resnet101_face-detection_cvpr22papermogface/summary) |
| 3 | [TinyMog人脸检测器-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_tinymog/summary) |
| 4 | [ULFD人脸检测模型-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_ulfd/summary) |
| 5 | [Mtcnn人脸检测关键点模型](https://modelscope.cn/models/damo/cv_manual_face-detection_mtcnn/summary) |
| 6 | [ULFD人脸检测模型-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_ulfd/summary) |


- 人脸识别

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [口罩人脸识别模型FaceMask](https://modelscope.cn/models/damo/cv_resnet_face-recognition_facemask/summary) |
| 2 | [口罩人脸识别模型FRFM-large](https://modelscope.cn/models/damo/cv_manual_face-recognition_frfm/summary) |
| 3 | [IR人脸识别模型FRIR](https://modelscope.cn/models/damo/cv_manual_face-recognition_frir/summary) |
| 4 | [ArcFace人脸识别模型](https://modelscope.cn/models/damo/cv_ir50_face-recognition_arcface/summary) |
| 5 | [IR人脸识别模型FRIR](https://modelscope.cn/models/damo/cv_manual_face-recognition_frir/summary) |

- 人脸活体识别

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [人脸活体检测模型-IR](https://modelscope.cn/models/damo/cv_manual_face-liveness_flir/summary) |
| 2 | [人脸活体检测模型-RGB](https://modelscope.cn/models/damo/cv_manual_face-liveness_flrgb/summary) |
| 3 | [静默人脸活体检测模型-炫彩](https://modelscope.cn/models/damo/cv_manual_face-liveness_flxc/summary) |

- 人脸关键点

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [FLCM人脸关键点置信度模型](https://modelscope.cn/models/damo/cv_manual_facial-landmark-confidence_flcm/summary) |

- 人脸属性 & 表情


| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [人脸表情识别模型FER](https://modelscope.cn/models/damo/cv_vgg19_facial-expression-recognition_fer/summary) |
| 2 | [人脸属性识别模型FairFace](https://modelscope.cn/models/damo/cv_resnet34_face-attribute-recognition_fairface/summary) |

## 来源说明
本模型及代码来自达摩院自研技术

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
```

