
# 文本指导的分割模型
- 此模型在文本分割数据集上进行训练，根据文本将图像中对应文本描述的物体分割出来。
- 模型在fss1000通用分割数据测试集上进行了zero shot测试。

## 模型结构信息
- 模型结构为Lseg结构，视觉encoder采用vit-large-patch16结构。
- 参考：https://github.com/isl-org/lang-seg
<img src="https://modelscope.cn/api/v1/models/damo/cv_vitl16_segmentation_text-driven-seg/repo?Revision=master&FilePath=langseg.jpg&View=true"
     alt="framework"
     style="width: 600px;" />

## 使用方式和范围

### 如何使用
- 直接推理，根据输入的文本描述对图像进行分割
- 用户可根据自身场景特点，收集自己的数据进行fine-tune

#### 代码范例
```python
# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = 'https://clip-multimodal.oss-cn-beijing.aliyuncs.com/xingguang/maas/data/text_driven_seg_demo.jpg'
test_input = {
            'image': input_location,
            'text': 'bear',
        }
model_id = 'damo/cv_vitl16_segmentation_text-driven-seg'
shop_seg = pipeline(Tasks.text_driven_segmentation, model=model_id)
result = shop_seg(test_input)
import cv2
# result[OutputKeys.MASKS] is segment map result,other keys are not used
cv2.imwrite('demo_textdrivenseg.jpg', result[OutputKeys.MASKS])

```
- 输入图像：
<img src="https://modelscope.cn/api/v1/models/damo/cv_vitl16_segmentation_text-driven-seg/repo?Revision=master&FilePath=demo.jpg&View=true"
     alt="input"
     style="width: 600px;" />
- 输入英文文本：bear
- 输出图像：
<img src="https://modelscope.cn/api/v1/models/damo/cv_vitl16_segmentation_text-driven-seg/repo?Revision=master&FilePath=demo_textdrivenseg.jpg&View=true"
     alt="output"
     style="width: 600px;" />


### 使用场景
- 适用于文本指导的语义分割

### 模型局限性以及可能的偏差
- 模型在通用场景数据集上训练，对于特定业务场景，比如商品效果较差，请用户自行评测后决定如何使用。
- 文本输入范围建议参考fss1000数据集的类别，对于其他文本输入可能效果较差。
- 由于是在粗标注数据集上进行的训练、因此模型精度建议结合具体场景评估后再使用。

## 训练数据
训练数据共约20w通用分割粗标注数据。

## 模型训练
### 预处理
- 图像输入：将长边resize到640分辨率，长宽比不变，对短边zero padding到640分辨率.
- 文本输入：英文文本描述要分割的物体，比如"bear"


## 数据评估及结果
该模型在fss1000数据测试集上zero shot iou: 85.14.

# 相关论文以及引用信息

```BibTeX
@inproceedings{
li2022languagedriven,
title={Language-driven Semantic Segmentation},
author={Boyi Li and Kilian Q Weinberger and Serge Belongie and Vladlen Koltun and Rene Ranftl},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=RriDjddCLN}
}
```