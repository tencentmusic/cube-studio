
# 读光文字检测
## News
- 2023年3月：
    - 新增DBNet训练/微调流程，支持自定义参数及数据集，详见代码示例。
- 2023年2月：
    - 新增业界主流[DBNet-通用场景](https://www.modelscope.cn/models/damo/cv_resnet18_ocr-detection-db-line-level_damo/summary)模型。

## 传送门
各场景文本检测模型：
- [SegLink++-通用场景行检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-line-level_damo/summary)
- [SegLink++-通用场景单词检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-word-level_damo/summary)
- [DBNet-通用场景行检测](https://www.modelscope.cn/models/damo/cv_resnet18_ocr-detection-db-line-level_damo/summary)

各场景文本识别模型：
- [ConvNextViT-手写场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/summary)
- [ConvNextViT-手写场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/summary)
- [ConvNextViT-文档印刷场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-document_damo/summary)
- [ConvNextViT-自然场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-scene_damo/summary)
- [ConvNextViT-车牌场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo/summary)
- [CRNN-通用场景](https://www.modelscope.cn/models/damo/cv_crnn_ocr-recognition-general_damo/summary)

整图OCR能力：
- [整图OCR-多场景](https://modelscope.cn/studios/damo/cv_ocr-text-spotting/summary)

欢迎使用！



## 模型描述

本模型是以自底向上的方式，先检测文本块和文字块之间的吸引排斥关系，然后对文本块聚类成行，最终输出单词的外接框的坐标值。ICGN模型介绍，详见：[Seglink++: Detecting dense and arbitrary-shaped scene text by instance-aware component grouping](https://www.researchgate.net/profile/Xiang-Bai/publication/334015431_Detecting_Dense_and_Arbitrary-shaped_Scene_Text_by_Instance-aware_Component_Grouping/links/5d2d79c9458515c11c337789/Detecting-Dense-and-Arbitrary-shaped-Scene-Text-by-Instance-aware-Component-Grouping.pdf)

![pipeline-icgn](https://modelscope.cn/api/v1/models/damo/cv_resnet18_ocr-detection-word-level_damo/repo?Revision=master&FilePath=./resources/pipeline-icgn.jpg&View=true)


## 期望模型使用方式以及适用范围
本模型主要用于给输入图片输出图中文字外接框坐标，具体地，模型输出的框的坐标为文字框四边形的四个角点的坐标，左上角为第一个点，按照顺时针的顺序依次输出各个点的坐标，分别为(x1,y1)(x2,y2)(x3,y3)(x4,y4)。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope之后即可使用ocr-detection的能力。

### 预处理和后处理
测试时的主要预处理和后处理如下：
- Resize Pad（预处理）: 输入图片长边resize到1024，短边等比例缩放，并且补pad到长短边相等
- threshold grouping（后处理）: node和link采用0.4和0.6的threshold，然后进行文字行grouping

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-word-level_damo')
result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg')
print(result)
```

## 数据评估及结果
模型在MLT17验证集上测试，结果如下

| Backbone |  Recall   | Precision |  F-score |
|:--------:|:---------:|:---------:|:--------:|
| ResNet18 |   74.8    |   85.3    |   79.7   |

以下为模型的一些可视化文字检测效果，检测框用绿色框表示。

![det-result-visu](https://modelscope.cn/api/v1/models/damo/cv_resnet18_ocr-detection-word-level_damo/repo?Revision=master&FilePath=./resources/det_result_visu.jpg&View=true)


### 模型局限性以及可能的偏差
- 模型是在特定英文数据集上训练的，在其他场景和语言的数据上有可能产生一定偏差，请用户自行评测后决定如何使用。
- 当前版本在python3.7环境CPU和单GPU环境测试通过，其他环境下可用性待测试

## 训练数据介绍
本模型训练数据集是MLT17/MLT19/IC15/TextOCR/HierText，训练数据数量约48K。

## 模型训练流程
本模型利用imagenet预训练参数进行初始化，然后在训练数据集上进行训练，先利用512x512尺度训练100epoch，然后在768x768尺度下finetune训练50epoch。

### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTex
@article{tang2019seglink++,
  title={Seglink++: Detecting dense and arbitrary-shaped scene text by instance-aware component grouping},
  author={Tang, Jun and Yang, Zhibo and Wang, Yongpan and Zheng, Qi and Xu, Yongchao and Bai, Xiang},
  journal={Pattern recognition},
  volume={96},
  pages={106954},
  year={2019},
  publisher={Elsevier}
}