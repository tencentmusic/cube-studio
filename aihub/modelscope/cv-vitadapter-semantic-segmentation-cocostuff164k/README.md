# vitadapter-image-semantic-segmentation模型介绍
- 本模型是Vision Transformer Adapter for Dense Predictions文章的简化版，更改的计算单元(C++编译模块更改为纯python实现)，方便部署。
- 给定一张输入图像，输出语义分割掩膜，类别，分数（虚拟分数）。

**其它相关模型体验**[Mask2Former-R50全景分割](https://www.modelscope.cn/models/damo/cv_r50_panoptic-segmentation_cocopan/summary) 

## 模型描述
本模型使用ViT-Adapter为backbone，Mask2Former为分割头。COCO-Stuff-164k数据库上训练。当前模型暂时不支持训练及finetune。

ViT-Adapter是一个对于稠密预测（例如检测，分割）任务十分友好的backbone模块。该方法包含一个基本backbone（ViT）和自适应的vitadapter。

其中，vitadpater包含了3个模块。第一个模块是Spatial Prior Module，用于从输入图像中提取空间特征。第二个模块是Spatial Feature Injector，用于将空间的先验注入到backbone（ViT）中，第三个模块是Multi-Scale Feature Extractor，用于从backbone（ViT）中提取分层特征。

![ViTadapter模型结构](description/vitadapter.png)

Mask2Former是一种能够解决任何图像分割任务（全景、实例或语义）的新架构。它包含了一个masked attention结构，通过将交叉注意力计算内来提取局部特征。

![Mask2Former模型结构](description/mask2former.png)

## 期望模型使用方式与适用范围
本模型适用范围较广，能对图片中包含的大部分感兴趣物体进行分割。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.utils.cv.image_utils import semantic_seg_masks_to_image
import cv2

segmentor = pipeline(Tasks.image_segmentation, model='damo/cv_vitadapter_semantic-segmentation_cocostuff164k')
input_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_semantic_segmentation.jpg'
result = segmentor(input_url)
draw_img = semantic_seg_masks_to_image(result[OutputKeys.MASKS])
cv2.imwrite('result.jpg', draw_img)
print("vitadapter DONE!")
```

#### 使用modelscope的数据库调用代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks,DownloadMode
from modelscope.utils.cv.image_utils import semantic_seg_masks_to_image
from modelscope.outputs import OutputKeys
from modelscope.msdatasets import MsDataset
import cv2

segmentor = pipeline(Tasks.image_segmentation, model='damo/cv_vitadapter_semantic-segmentation_cocostuff164k')
ms_ds_val = MsDataset.load("COCO_segmentation_inference", namespace="modelscope", split="validation", download_mode=DownloadMode.FORCE_REDOWNLOAD)
result = segmentor(ms_ds_val[0]["InputImage:FILE"])
draw_img = semantic_seg_masks_to_image(result[OutputKeys.MASKS])
cv2.imwrite('result.jpg', draw_img)
print("vitadapter DONE!")
```


### 模型局限性以及可能的偏差
- 部分非常规图片或感兴趣物体占比太小或遮挡严重可能会影响分割结果
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试
## 训练数据介绍
- [COCO-Stuff-164k](https://github.com/nightrome/cocostuff)：包含118k张训练图片，共171个类别。
### 预处理
测试时主要的预处理如下：
- Resize：先将原始图片的短边Resize到896，等比例缩放。此时如果长边超过了3584，则按照最长边为3584，重新计算Resize的scale进行Resize，滑动测试
- Normalize：图像归一化，减均值除以标准差

## 数据评估及结果
| Backbone |  Pretrain   | aAcc    | mIoU     | mAcc    | 
|:--------:|:-----------:|:-------:|:--------:|:-------:|
|vitadapter|    BEiT-L   |  69.91  |   47.21  |  57.17  | 
## 引用
```BibTeX
@article{chen2022vision,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```