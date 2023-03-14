
# Cycle-CenterNet表格结构识别模型介绍
表格结构识别，即给定一张图片，检测出图中单元格的物理坐标（四个顶点）以及逻辑坐标（行号列号）。

## 模型描述

本模型是以自底向上的方式: 1）基于单元格中心点回归出到4个顶点的距离，解码出单元格bbox；同时基于单元格顶点，回归出到共用该顶点的单元格的中心点距离，解码出gbox。2）基于gbox(group box)，将离散的bbox拼接起来得到精准完整的电子表格；3）第二步的拼接将单元格从“离散”变为“连续”，因此用后处理算法获得单元格的行列信息。目前上线模型实现前两步的功能，第三步暂时未提供。Cycle-CenterNet模型介绍，详见：[Parsing Table Structures in the Wild](https://openaccess.thecvf.com/content/ICCV2021/papers/Long_Parsing_Table_Structures_in_the_Wild_ICCV_2021_paper.pdf) 。

![pipeline](./description/Pipeline.jpg)


## 期望模型使用方式以及适用范围
本模型主要用于给输入图片输出图中表格单元格拼接后的物理坐标，具体地，模型输出的框的坐标为单元格的四个角点的坐标，左上角为第一个点，按照顺时针的顺序依次输出各个点的坐标，分别为(x1,y1)(x2,y2)(x3,y3)(x4,y4)。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope之后即可使用table-recognition的能力。

### 预处理和后处理
测试时的主要预处理和后处理如下：
- Resize Pad（预处理）: 输入图片长边resize到1024，短边等比例缩放，并且补pad到长短边相等。同时有减均值除方差等归一化操作。
- 表格拼接(后处理)：基于模型推理的gbox，将离散bbox拼接起来得到完整表格。

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
table_recognition = pipeline(Tasks.table_recognition, model='damo/cv_dla34_table-structure-recognition_cycle-centernet')
result = table_recognition('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/table_recognition.jpg')
print(result)
```


### 模型局限性以及可能的偏差
- 模型主要用于有线表格，无线表格不支持

## 训练数据介绍
本模型训练数据为WTW，训练集为10970张。

## 模型训练流程
本模型利用imagenet预训练参数进行初始化，然后在训练数据集上进行训练。


### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTex
@inproceedings{long2021parsing,
  title={Parsing table structures in the wild},
  author={Long, Rujiao and Wang, Wen and Xue, Nan and Gao, Feiyu and Yang, Zhibo and Wang, Yongpan and Xia, Gui-Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={944--952},
  year={2021}
}