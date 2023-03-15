# Bad Image Detecting

## 模型描述
基于mobilenet-v2的一个简单基线，可以有效检测异常图像，包括编解码或者图像宽高、行偏移错误等造成的花屏，绿屏图像。

| <img src="./data/demo_resize.png"  width=1024 alt="demo for bad image detecting model">|
| :-----------------------------------------------------------------------------------: |
|                                     Bad Image Detecting                                           |

## 期望模型使用方式以及适用范围
本模型适用于检测图像/视频中的坏帧，包括花屏，绿屏等异常帧，输出图像检测类型，包含[花屏， 绿屏， 正常]。模型适用于1080P及以下分辨率图像质量评价。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg'
test_pipeline = pipeline(Tasks.bad_image_detecting, 'damo/cv_mobilenet-v2_bad-image-detecting')
result = test_pipeline(img)
print(result)
```

## 验证数据介绍
cv_mobilenet-v2_bad-image-detecting_validation sub

包含正常图像，花屏图像及绿屏图像。数据使用自有视频/图像数据经过编解码、宽高偏置错误或者搜集得到。图像标签0，1，2分别代表正常图像、花屏图像及绿屏图像.

文件类型：.PNG

文件数量：252



## 数据评估及结果
| Dataset | ACCURACY |
|:---- |:---- |
|cv_mobilenet-v2_bad-image-detecting_validation|0.9921|

```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.bad_image_detecting import \
    BadImageDetectingDataset


tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model_id = 'damo/cv_mobilenet-v2_bad-image-detecting'
cache_path = snapshot_download(model_id)
config = Config.from_file(os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_val = MsDataset.load(
    'cv_mobilenet-v2_bad-image-detecting_validation',
    namespace='charlesHuang',
    subset_name='subset',
    split='train',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

eval_dataset = BadImageDetectingDataset(dataset_val, config.dataset)
kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)

trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()

print(metric_values)



#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_mobilenet-v2_bad-image-detecting.git
```

### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@misc{
      title={Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation}, 
      author={Mark Sandler Andrew Howard},
      year={2019},
      eprint={2111.07104},
      archivePrefix={arXiv}
}