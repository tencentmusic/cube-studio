
# LORE无线表格结构识别模型介绍
表格结构识别，即给定一张图片，检测出图中单元格的物理坐标（四个顶点）以及逻辑坐标（行号列号）。在无线表格中，单元格的物理坐标使用表格内文字的外接框。

## 模型描述

本模型的主要原理为: 1）基于无线单元格中心点回归出到4个顶点的距离，解码出单元格bbox；2）结合视觉特征与单元格bbox信息，采用两个级联回归器兼顾全局与局部注意力，直接对单元格的逻辑坐标进行回归；3）模型训练时显式利用单元格间与单元格内逻辑约束对模型进行优化。详情可访问[论文“LORE: Logical Location Regression Network for Table Structure Recognition”](https://arxiv.org/abs/2303.03730)与[开源项目](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/LORE-TSR)。

![pipeline](./description/Pipeline.png)


## 期望模型使用方式以及适用范围
本模型预期的输入为截取好的单个无线表格图片，如果图中含有非表格内容或多个表格会导致结果错误。模型的输出为图中无线表格单元格的物理坐标与逻辑坐标，具体地，物理坐标为单元格的四个角点的坐标，左上角为第一个点，按照顺时针的顺序依次输出各个点的坐标，分别为(x1,y1)(x2,y2)(x3,y3)(x4,y4)，逻辑坐标为从0开始的起始及结束行列号，具体格式为(start_row,end_row,start_column,end_column)。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope之后即可使用lineless-table-recognition的能力。

### 预处理和后处理
测试时的主要预处理和后处理如下：
- Resize Pad（预处理）: 输入图片长边resize到768，短边等比例缩放，并且补pad到长短边相等。同时有减均值除方差等归一化操作。
- 无后处理。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
lineless_table_recognition = pipeline(Tasks.lineless_table_recognition, model='damo/cv_resnet-transformer_table-structure-recognition_lore')
result = lineless_table_recognition('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/lineless_table_recognition.jpg')
print(result)
```


### 模型局限性以及可能的偏差
- 模型主要用于无线表格，有线表格不支持

## 训练数据介绍
本模型训练数据部分来自SciTSR与PubTabNet，训练集共45000张。

## 模型训练流程
本模型利用imagenet预训练参数进行初始化，然后在训练数据集上进行训练。
