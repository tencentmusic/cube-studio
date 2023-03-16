
# 细粒度动物识别（8k类）模型介绍

本模型是对含有动物的图像进行标签识别，无需任何额外输入，输出动物的类别标签，目前已经覆盖了8K多类的细粒度的动物类别。

## 模型描述
模型采用resnest101网络结构。

## 使用方式和范围

使用方式：
- 直接推理，对输入的图像，输入图像直接进行推理。

使用场景:
- 适合含有动物的图像进行动物标签识别，期望图像中动物占比不要过小。

代码范例:

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

animal_recognition= pipeline(
            Tasks.animal_recognition,
            model='damo/cv_resnest101_animal_recognition')
result = animal_recognition('https://pailitao-image-recog.oss-cn-zhangjiakou.aliyuncs.com/mufan/img_data/maas_test_data/dog.png')
print(result)
```

## 训练数据
训练数据共约700w带有动物标签的数据。

## 模型训练
### 预处理
图像输入：resize到256\*256，然后CenterCrop到224*224。

### LR scheduler
初始LR为 0.0003，每隔10个epoch，lr调整为原来的1/4，共训练100个epoch。

## 数据评估及结果
通过收集线上的实际应用数据进行评测精度为72.1%。
