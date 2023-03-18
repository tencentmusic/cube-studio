# swinL-image-panoptic-segmentation模型介绍
给定一张输入图像，输出全景分割掩膜，类别，分数（虚拟分数）。

全景分割是要分割出图像中的stuff，things。stuff是指天空，草地等不规则区域，things是指可数的物体，例如人，车，猫等。

**请前往**[Mask2Former-R50全景分割](https://www.modelscope.cn/models/damo/cv_r50_panoptic-segmentation_cocopan/summary) **体验模型finetune**

## 模型描述
本模型使用swin large为backbone，Mask2Former为分割头。COCO全景分割数据库上训练。

Swin transformer是一种具有金字塔结构的transformer架构，其表征通过shifted windows计算。Shifted windows方案将自注意力的计算限制在不重叠的局部窗口上，同时还允许跨窗口连接，从而带来更高的计算效率。分层的金字塔架构则让其具有在各种尺度上建模的灵活性。这些特性使swin transformer与广泛的视觉任务兼容，并在密集预测任务如COCO实例分割上达到SOTA性能。其结构如下图所示。

![Swin模型结构](description/swin.png)

Mask2Former是一种能够解决任何图像分割任务（全景、实例或语义）的新架构。它包含了一个masked attention结构，通过将交叉注意力计算内来提取局部特征。

![Mask2Former模型结构](description/mask2former.png)

## 期望模型使用方式与适用范围
本模型适用范围较广，能对图片中包含的大部分感兴趣物体（COCO things 80类，stuff 53类）进行分割。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import panoptic_seg_masks_to_image
from modelscope.outputs import OutputKeys
import cv2

segmentor = pipeline(Tasks.image_segmentation, model='damo/cv_swinL_panoptic-segmentation_cocopan')
input_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_panoptic_segmentation.jpg'
result = segmentor(input_url)
draw_img = panoptic_seg_masks_to_image(result[OutputKeys.MASKS])
cv2.imwrite('result.jpg', draw_img)
```

#### 使用modelscope的数据库调用代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks,DownloadMode
from modelscope.utils.cv.image_utils import panoptic_seg_masks_to_image
from modelscope.outputs import OutputKeys
from modelscope.msdatasets import MsDataset
import cv2

segmentor = pipeline(Tasks.image_segmentation, model='damo/cv_swinL_panoptic-segmentation_cocopan')
ms_ds_val = MsDataset.load("COCO_segmentation_inference", namespace="modelscope", split="validation", download_mode=DownloadMode.FORCE_REDOWNLOAD)
result = segmentor(ms_ds_val[0]["InputImage:FILE"])
draw_img = panoptic_seg_masks_to_image(result[OutputKeys.MASKS])
cv2.imwrite('result.jpg', draw_img)
```

### 模型局限性以及可能的偏差
- 部分非常规图片或感兴趣物体占比太小或遮挡严重可能会影响分割结果
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试
## 训练数据介绍
- [COCO-panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) ：COCO全称是Common Objects in Context，是Microsoft团队提供的一个可以用来图像识别、检测和分割的数据集。COCO-panoptic 2017 包含前景（things）80个类别，背景（stuff）53个类别。
### 预处理
测试时主要的预处理如下：
- Resize：先将原始图片的短边Resize到800，等比例缩放。此时如果长边超过了1333，则按照最长边为1333，重新计算Resize的scale进行Resize
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

## 数据评估及结果
| Backbone |  Pretrain   | box mAP | mask mAP | PQ | 
|:--------:|:-----------:|:-------:|:--------:|:-------:|
|  Swin-L  | ImageNet-21K |  52.2   |   48.5   |  57.6   | 

## 引用
```BibTeX
@inproceedings{cheng2022masked,
  title={Masked-attention mask transformer for universal image segmentation},
  author={Cheng, Bowen and Misra, Ishan and Schwing, Alexander G and Kirillov, Alexander and Girdhar, Rohit},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1290--1299},
  year={2022}
}
```
```BibTeX
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```