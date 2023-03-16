# 通用识别介绍

本模型是对包含主体物体的图像进行标签识别，无需任何额外输入，输出主体物体的类别标签，目前已经覆盖了5W多类的物体类别。

## 模型描述
模型采用resnest101结构
## 使用方式和范围

使用方式：
- 直接推理，对输入的图像直接进行推理

使用场景:
- 适合含有主体物体的图像进行物体标签识别，期望图像中主体物体占比不要过小

代码范例:

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

general_recognition = pipeline(
            Tasks.general_recognition,
            model='damo/cv_resnest101_general_recognition')
result = general_recognition('https://pailitao-image-recog.oss-cn-zhangjiakou.aliyuncs.com/mufan/img_data/maas_test_data/dog.png')

```


## 训练数据
训练数据共约3000w带有各类物体标签的数据

## 模型训练
### 预处理
--图像输入：resize到256\*256，然后CenterCrop到224*224

### LR scheduler
初始LR为 0.0003，每隔10个epoch，lr调整为原来的1/4，共训练100个epoch。
## 数据评估及结果
通过收集线上的实际应用数据进行评测精度为80.1%