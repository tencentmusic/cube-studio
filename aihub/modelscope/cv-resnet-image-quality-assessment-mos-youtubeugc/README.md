# Image Quality Assessment for UGC

## 模型描述
基于resnet的一个简单基线，可以有效评估图像的无参考画质，达到SOTA性能。其网络结构如下图所示：

<img src="./data/net.png" width=512 alt="Net architecture">

| <img src="./data/demo_resize.png"  width=1024 alt="demo for image quality assessment model">|
| :-----------------------------------------------------------------------------------: |
|                                     image quality assessment                                           |



## 期望模型使用方式以及适用范围
本模型适用于UGC图像的视觉质量评价，输出评价mos分,范围[0, 1],值越大代表图像质量越好。模型适用于1080P及以下分辨率图像质量评价。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg'
image_quality_assessment_pipeline = pipeline(Tasks.image_quality_assessment_mos, 'damo/cv_resnet_image-quality-assessment-mos_youtubeUGC')
result = image_quality_assessment_pipeline(img)[OutputKeys.SCORE]
print(result)
```

### 模型局限性以及可能的偏差
由于训练数据为YouTube UGC Dataset，针对实拍ugc类的图像评价结果良好，而其他类型图像可能表现不佳。

## 验证数据介绍
YouTube UGC Dataset Validation sub

随机选择YouTube UGC Dataset中20%视频，每个视频抽取一帧生成验证输入图像，mos分采用视频对应值。

文件类型：.PNG

文件数量：221

内容：每幅图来自不同的视频，进模型前会进行前处理，mos分标签归一化到[0,1]


## 数据评估及结果
| Dataset | PLCC | SRCC | RMSE |
|:---- |:----    |:---- |:----|
|YouTube UGC|0.8219|0.8224|0.0724|

```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.image_quality_assmessment_mos import \
    ImageQualityAssessmentMosDataset


tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model_id = 'damo/cv_resnet_image-quality-assessment-mos_youtubeUGC'
cache_path = snapshot_download(model_id)
config = Config.from_file(os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_val = MsDataset.load(
    'vqa_mos_youtubeUGC_validation',
    namespace='charlesHuang',
    subset_name='subset',
    split='train',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

eval_dataset = ImageQualityAssessmentMosDataset(dataset_val, config.dataset)
kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)

trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()

print(metric_values)

```
#### Clone with HTTP
```bash
 git clone  https://www.modelscope.cn/damo/cv_resnet_image-quality-assessment-mos_youtubeUGC.git
```

### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@misc{wen2021strong,
      title={A strong baseline for image and video quality assessment}, 
      author={Shaoguo Wen and Junle Wang},
      year={2021},
      eprint={2111.07104},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
