
# vit-object-detection模型介绍
Exploring Plain Vision Transformer Backbones for Object Detection文章复现，采用COCO数据集训练。

<img src=resources/test_object_1_rst.jpg width=35% /><img src=resources/test_object_2_rst.jpg width=35% />
## 期望模型使用方式与适用范围
本模型适用范围较广，能对图片中包含的大部分前景物体（COCO 80类）进行定位。
### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
object_detect = pipeline(Tasks.image_object_detection,model='damo/cv_vit_object-detection_coco')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'
result = object_detect(img_path)
print(result)
```
### 模型局限性以及可能的偏差
- 使用coco2017对论文复现，指标略逊于论文指标（0.002）。
- 召回目标coco-80类数据范围。
## 训练数据介绍
- coco2017数据集,Microsoft团队提供的一个可以用来图像识别、检测和分割的数据集。COCO2017包含训练集118287张、验证集5000张、测试集40670张，共有80类物体。具体可见(https://cocodataset.org/#detection-2017)
## 模型训练流程
- 模型训练
  使用ImageNet-1K上的预训练模型VitBase作为基础backbone，采取MaskRCNN结构复现论文结果。
### 预处理
- 给定一张输入图像，分辨率归一化至(1024,1024),颜色值减均值除方差归一化处理。
## 数据评估及结果
| Backbone |  Pretrain   | box mAP | mask mAP |   Remark   |
|:--------:|:-----------:|:-------:|:--------:| ---------- |
| ViT-Base | ImageNet-1k |  51.6   |   45.9   | [official](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) |
| ViT-Base | ImageNet-1k |  51.1   |   45.5   | [unofficial](https://github.com/ViTAE-Transformer/ViTDet) |  
| ViT-Base | ImageNet-1k |  51.4   |   45.7   | modelscope |
## 引用
```BibTeX
@article{Li2022ExploringPV,
  title={Exploring Plain Vision Transformer Backbones for Object Detection},
  author={Yanghao Li and Hanzi Mao and Ross B. Girshick and Kaiming He},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.16527}
}
```
