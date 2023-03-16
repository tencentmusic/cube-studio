
# 图像画质损伤分析介绍
图像画质损伤分析模型分析输入图像，输出常见画质损伤的各维度客观评分，包括清晰度评估、点状噪声水平评估、压缩噪声水平评估。

## 模型描述
采用resnet50结构，使用图片网站 [pexels](www.pexels.com) 中最受欢迎的 130,000 张图片作为训练集，通过模拟各种降质来训练模型对清晰度、点状噪声、压缩噪声的评估。

## 期望模型使用方式以及适用范围
对输入的图像直接进行推理，或对视频抽帧结果进行处理得到整个视频的画质损伤分析结果。

### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg'
image_quality_assessment_degradation_pipeline = pipeline(Tasks.image_quality_assessment_degradation, 'damo/cv_resnet50_image-quality-assessment_degradation', model_revision='v1.0.0')
result = image_quality_assessment_degradation_pipeline(img)
print(dict(zip(result[OutputKeys.LABELS], result[OutputKeys.SCORES])))
```


### 模型局限性以及可能的偏差
支持对各类常见模糊及噪声、压缩噪声的程度评估。
## 测试数据介绍
KADID-10k

文件类型：.PNG

文件数量：81组,10206张图片

内容：每组图片包含原图及25种模拟损伤图像
## 数据评估及结果
| Dataset | PLCC | SRCC |
|:------- |:---- |:---- |
|KADIDS-10k|0.9654|0.9963|

### 评测方式说明
测试每张图片各类相关损伤1-5级画质损伤预测结果，计算PLCC及SRCC，并对所有图像结果取平均

### 评测代码
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.image_quality_assessment_degradation import ImageQualityAssessmentDegradationDataset


tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model_id = 'damo/cv_resnet50_image-quality-assessment_degradation'
cache_path = snapshot_download(model_id)
config = Config.from_file(os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_val = MsDataset.load(
    'KADID-10k-database',
    namespace='damo',
    subset_name='default',
    split='validation',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

eval_dataset = ImageQualityAssessmentDegradationDataset(dataset_val)
kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)

trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()

print(metric_values)
```

## 引用
如果你觉得该模型有所帮助，请考虑引用下面的相关的论文：
```
@inproceedings{wang2021rich,
  title={Rich features for perceptual quality assessment of UGC videos},
  author={Wang, Yilin and Ke, Junjie and Talebi, Hossein and Yim, Joong Gon and Birkbeck, Neil and Adsumilli, Balu and Milanfar, Peyman and Yang, Feng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13435--13444},
  year={2021}
}
```