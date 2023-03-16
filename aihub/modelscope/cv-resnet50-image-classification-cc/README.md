
# 内容审核模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[图片智能鉴黄](https://vision.aliyun.com/experience/detail?tagName=imageaudit&children=IdentifyPorn&spm=a2cio.27993362)。

内容审核模型-鉴黄

## 模型描述
模型([项目地址](https://github.com/emiliantolo/pytorch_nsfw_model))采用resnet50作为backbone，训练数据集为NSFW，本模型使用去噪后的大约250k训练数据（40g左右），训练集包含五类标签，分别为pornography images，hentai images and pornographic drawings，sexually explicit images, safe for work neutral images，safe for work drawings and anime。其中sexually explicit images认为是不合规的内容，其余均为合规内容。

## 模型效果
![CC模型效果](result.png)

## 模型使用方式和使用范围
本模型可以用来判断图片中的内容是否涉黄，分数越高则不涉黄的可能性越高。

### 使用方式
- 推理：输入图片，返回图片不涉黄的可能性。

### 目标场景
鉴黄模型是贯彻《互联网信息服务管理办法》、促进网络环境绿色健康化发展以及降低人力成本，提升鉴黄效率的一项重要举措。鉴黄模型可以使用在大型网站服务器、各类图片云服务器以及其他支持图片存储的社交应用、垂直社区等UGC 平台上，这些平台都需要对用户上传的图像内容进行审核。

### 模型局限性及可能偏差
- 目前模型只支持对设计色情图片的审核，对血腥，暴力审核场景暂不支持

### 预处理
测试时主要的预处理如下：
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数



### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

content_check_func = pipeline(Tasks.image_classification, 'damo/cv_resnet50_image-classification_cc')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/content_check.jpg'
result = content_check_func(img_path)
print(f'content check output: {result}.')
```


## 来源说明
本模型及代码来自第三方开源技术([项目地址](https://github.com/emiliantolo/pytorch_nsfw_model))。

