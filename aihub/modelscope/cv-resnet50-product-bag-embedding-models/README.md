
# 商品图像特征模型介绍

本模型是对商品图像进行表征向量提取，用户可基于表征向量进行大规模的同款/相似款商品搜索；无需额外输入，模型可自动进行箱包商品的主体抠图，并基于主体提取结果完成表征向量提取。

## 模型描述
整体模型分为两阶段，第一阶段为抠图模型预处理，负责将图片中的主体目标进行提取，基于提取的目标进行表征提取；抠图模型采用yolox模型。
第二阶段为表征模型，表征模型采用resnet50网络结构，在resnet50最后一个卷积层之后，接入全局平均池化，并通过全连接层，得到512维度表征向量。

## 使用方式和范围

使用方式：
- 直接推理，对输入的图像，自动完成主体抠图，并基于抠图结果进行表征提取。

使用场景:
- 适合大规模箱包类目的商品表征向量提取，用户基于表征向量可完成大规模同款/相似款图片搜索。

代码范例:

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

product_embedding = pipeline(
            Tasks.product_retrieval_embedding,
            model='damo/cv_resnet50_product-bag-embedding-models')
result = product_embedding('https://mmsearch.oss-cn-zhangjiakou.aliyuncs.com/maas_test_img/tb_image_share_1666002161794.jpg')
```

## 训练数据
训练数据为千万级别的Triplet三元组; 先对每张图片做抠图预处理；基于抠图的结果，利用TripletLoss进行训练。

## 模型训练
### 预处理
图像输入：原始图像resize到416*416并做检测抠图预处理，抠图结果resize到224x224，输入表征模型进行特征提取。

### LR scheduler
表征模型训练过程，初始LR为 0.001，每隔20个epoch，lr调整为原来的1/10，共训练60个epoch。

## 数据评估及结果
通过收集线上的实际应用数据进行评测Top1同款率为70.9%。