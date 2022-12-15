# 模型名称 模型版本

模型的描述或者引用来源。

## 模型描述

这里添加一些模型的详细描述，效果图，算法结构等，让用户详细了解

## 使用和限制

您可以使用原始模型进行图像分类。查看模型中心以查找针对您感兴趣的任务的微调版本。

### 如何使用

python的示例代码，例如下面

```python
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

