# resnet18-human-detection模型介绍
给定一张输入图像，输出图像中人体的坐标。

<img src=resources/test_object_0_rst.jpg width=25% /><img src=resources/test_object_1_rst.jpg width=25% />

## 期望模型使用方式与适用范围
本模型适用范围较广，覆盖室内外、监控、单人多人等大部分场景。
### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
object_detect = pipeline(Tasks.human_detection,model='damo/cv_resnet18_human-detection')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'
result = object_detect(img_path)
print(result)
```
### 模型局限性以及可能的偏差
- 目前覆盖多种场景，在小目标或者遮挡区域可能存在误检
## 训练数据介绍
- coco2017数据集，Microsoft团队提供的一个可以用来图像识别、检测和分割的数据集。COCO2017包含训练集118287张、验证集5000张、测试集40670张，共有80类物体，详情见：https://cocodataset.org/#detection-2017
- object365数据集，北京旷视科技有限公司与北京智源人工智能研究院共同发布，该数据集总共包含63万张图像，覆盖365个类别，高达1000万框数，规模大、质量高、泛化能力强，详情见：http://www.objects365.org/overview.html
- 互联网爬取数据，三种数据集抽离行人数据集合训练
## 模型训练流程
- 模型训练
  FasterRCNN-Pipline，Resnet18为基础BackBone，引入DyHead模块提升特征尺度、空间位置、通道感知层面注意力。
### 预处理
- 分辨率归一化(1333,800)，首先按照短边800等比缩放，若长边大于1333，则按照长边1333缩放；
- 颜色值减均值除方差归一化
- 图像pad至32的倍数
## 数据评估及结果
模型在COCO2017val-human子集上进行测试,mAP为59.8%

## 相关链接
若希望使用稳定高并发的API服务，可使用视觉智能开放平台对应的API：[人体检测API](https://vision.aliyun.com/experience/detail?spm=a2cvz.27717767.J_7524944390.39.66cd28d0bABaxq&tagName=facebody&children=DetectPedestrian)。

## 引用
```BibTeX
@InProceedings{Dai_2021_CVPR,
    author    = {Dai, Xiyang and Chen, Yinpeng and Xiao, Bin and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Zhang, Lei},
    title     = {Dynamic Head: Unifying Object Detection Heads With Attentions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7373-7382}
}
@inproceedings{renNIPS15fasterrcnn,
    Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
    Title = {Faster {R-CNN}: Towards Real-Time Object Detection
             with Region Proposal Networks},
    Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
    Year = {2015}
}
```
