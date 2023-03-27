
# 商品显著性分割模型
- 此模型在商品显著性分割数据集上进行训练，主要对商品图像进行显著性分割。
- 模型在fss1000通用分割数据集上进行了zero shot测试。

## 模型结构信息
- 模型结构为Denseclip结构，视觉encoder采用vit-base-patch16结构。
- 参考：https://github.com/raoyongming/DenseCLIP
<img src="https://modelscope.cn/api/v1/models/damo/cv_vitb16_segmentation_shop-seg/repo?Revision=master&FilePath=denseclip.jpg&View=true"
     alt="framework"
     style="width: 600px;" />

## 使用方式和范围

### 如何使用
- 需要自行安装opencv, mmcv-full，注意版本问题，pyarrow库建议使用8.0.0版本
- 直接推理，对输入的商品图像进行显著性区域提取
- 用户可根据自身场景特点，收集自己的数据进行fine-tune

#### 代码范例
```python
# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = 'https://clip-multimodal.oss-cn-beijing.aliyuncs.com/xingguang/maas/data/shop_seg_demo.jpg'
model_id = 'damo/cv_vitb16_segmentation_shop-seg'
shop_seg = pipeline(Tasks.shop_segmentation, model=model_id)
result = shop_seg(input_location)
import cv2
# result[OutputKeys.MASKS] is shop segment map result,other keys are not used
cv2.imwrite('demo_shopseg.jpg', result[OutputKeys.MASKS])
```
- 输入图像：
<img src="https://modelscope.cn/api/v1/models/damo/cv_vitb16_segmentation_shop-seg/repo?Revision=master&FilePath=demo.jpg&View=true"
     alt="input"
     style="width: 600px;" />
- 输出图像：
<img src="https://modelscope.cn/api/v1/models/damo/cv_vitb16_segmentation_shop-seg/repo?Revision=master&FilePath=demo_shopseg.jpg&View=true"
     alt="output"
     style="width: 600px;" />

### 使用场景
- 适用于商品显著性分割

### 模型局限性以及可能的偏差
模型在商品显著性分割数据集上训练，主要针对商品场景，对通用场景效果较差，请用户自行评测后决定如何使用。

## 训练数据
训练数据共约40w商品显著性分割数据。

## 模型训练
### 预处理
- 图像输入：将长边resize到1024分辨率，长宽比不变，对短边zero padding到1024分辨率.


## 数据评估及结果
该模型在fss1000全量数据集上（1万张图像）zero shot iou: 79.12

## 相关论文以及引用信息

```BibTeX
@inproceedings{rao2021denseclip,
  title={DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting},
  author={Rao, Yongming and Zhao, Wenliang and Chen, Guangyi and Tang, Yansong and Zhu, Zheng and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```