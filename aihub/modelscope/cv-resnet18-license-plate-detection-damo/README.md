
# 车牌检测模型介绍
给定一张图片，检测出图中车牌的位置并输出车的类型（比如小汽车，挂车，新能源车等）。

## 模型描述

本模型是以自底向上的方式: 1）首先识别出车牌的中心点；2）基于中心点回归出车牌的bbox；3）基于车牌中心点识别出车类型。模型介绍，详见：

![pipeline](https://modelscope.cn/api/v1/models/damo/cv_resnet18_license-plate-detection_damo/repo?Revision=master&FilePath=./description/Pipeline.jpg&View=true)


## 期望模型使用方式以及适用范围
输入图片，模型自动检测出所有车牌并给出对应的车型。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope之后即可使用license-plate-detection的能力。

### 预处理和后处理
测试时的主要预处理和后处理如下：
- Resize Pad（预处理）: 输入图片长边resize到1024，短边等比例缩放，并且补pad到长短边相等。同时有减均值除方差等归一化操作。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
license_plate_detection = pipeline(Tasks.license_plate_detection, model='damo/cv_resnet18_license-plate-detection_damo')
result = license_plate_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/license_plate_detection.jpg')
print(result)
```


### 模型局限性以及可能的偏差
- 由于除小汽车以外，其他车型数据很少，因此车型识别效果欠佳。

## 模型训练流程
本模型利用imagenet预训练参数进行初始化，然后在训练数据集上进行训练。