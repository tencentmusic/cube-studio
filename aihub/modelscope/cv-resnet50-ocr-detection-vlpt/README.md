
# VLPT多模态文字检测预训练模型介绍
文字检测天然涉及到图像和文本两种模态，VLPT通过设计三个图像特征和文本特征相互交互的预训练
任务，使得模型backbone具有了优秀的文字感知能力。该backbone参数可以作为大多数文字检测模型
的初始化参数，得到更好的训练效果。

## 模型描述
本模型主要包含三个特征编码器，分别为图像特征编码器、文本特征编码器以及图文交互编码器。通过
三个图文交互预训练任务（图文对比、文本掩码建模和文本存在判断），图像特征编码器能有效区分文本与非文本区域，该特征编码器可以无缝替代主流文字检测模型的backbone部分，本文以[DB](https://arxiv.org/pdf/1911.08947.pdf)检测方法为例。
VLPT模型介绍，详见：[Vision-Language Pre-Training for Boosting Scene Text Detectors](https://openaccess.thecvf.com/content/CVPR2022/papers/Song_Vision-Language_Pre-Training_for_Boosting_Scene_Text_Detectors_CVPR_2022_paper.pdf) 。

![pipeline](./resources/pipeline.png)


## 期望模型使用方式以及适用范围
本模型主要用于给输入图片输出图中文字外接框坐标，具体地，模型输出的框的坐标为文字框多边形的N个角点的坐标，分别为(x1,y1)(x2,y2)(x3,y3)...(xn,yn)。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。


### 如何使用
在安装完成ModelScope之后即可使用ocr-detection-vlpt的能力。

### 预处理和后处理
测试时的主要预处理和后处理如下：
- Resize（预处理）: 输入图片短边resize到736，长边等比例缩放，同时有减均值除方差等归一化操作。
- threshold（后处理）: 二值化阈值为0.3，检测框置信度阈值为0.5。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet50_ocr-detection-vlpt')
result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection_vlpt.jpg')
print(result)
```


### 模型局限性以及可能的偏差
- 模型主要用于英文单词检测，中文暂不支持。

## 训练数据介绍
本模型预训练数据为synthtext，训练集为80w张，后在totaltext上finetune，数据量为1255张。

## 模型训练流程
本模型利用imagenet预训练参数进行初始化，然后在预训练训练数据集上进行预训练训练，最后在相应数据集上进行finetune。

## 数据评估及结果
以下表格为totaltext数据集上的评测结果，baseline与我们的模型均未使用deformable conv。
| 模型          | precision | recall | fmeasure |
| --------------- | --------- | -------- | ------------ |
| DB_wo_dconv | 0.85    | 0.79   | 0.82           |
| DB_wo_dconv (VLPT)  | 0.88    | 0.82   | 0.85      |



### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTex
@inproceedings{song2022vision,
  title={Vision-Language Pre-Training for Boosting Scene Text Detectors},
  author={Song, Sibo and Wan, Jianqiang and Yang, Zhibo and Tang, Jun and Cheng, Wenqing and Bai, Xiang and Yao, Cong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15681--15691},
  year={2022}
}
