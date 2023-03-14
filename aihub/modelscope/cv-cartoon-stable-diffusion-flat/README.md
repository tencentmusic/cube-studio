
# 卡通系列文生图模型-扁平风

输入一段文本提示词，实现特定风格卡通图像生成，返回符合文本描述的扁平风格结果图像。

其生成效果如下所示：

(1) 人物生成

![生成效果](description/demo.png)

(2) 物体场景生成

![生成效果](description/demo1.png)

人物文本提示词：‘sks style, a portrait painting of [name]’

物体场景文本提示词：‘sks style, a painting of [name]’


## 模型描述

该模型通过在最新的文生图模型Stable-Diffusion-2.1上执行卡通风格微调实现，通过在文本提示词中加入‘sks style'生成对应风格符合文本描述的图像结果。


## 使用方式和范围

使用方式：
- 支持16G及以上GPU推理，输入包含‘sks style'的文本提示词进行推理;

目标场景:
- 艺术创作、社交娱乐、隐私保护场景，自动化生成名人卡通肖像。

### 如何使用

在ModelScope框架上，输入包含‘sks style'的文本提示词，即可以通过简单的Pipeline调用来使用卡通风格文生图模型。

#### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipe = pipeline(Tasks.text_to_image_synthesis, model='damo/cv_cartoon_stable_diffusion_flat', model_revision='v1.0.0')
output = pipe({'text': 'sks style, a portrait painting of Johnny Depp'})
cv2.imwrite('result.png', output['output_imgs'][0])
print('Image saved to result.png')

# 更佳实践
pipe = pipeline(Tasks.text_to_image_synthesis, model='damo/cv_cartoon_stable_diffusion_flat', model_revision='v1.0.0')
from diffusers.schedulers import EulerAncestralDiscreteScheduler
pipe.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipeline.scheduler.config)
output = pipe({'text': 'sks style, a portrait painting of Johnny Depp'})
cv2.imwrite('result.png', output['output_imgs'][0])
print('Image saved to result.png')

print('finished!')

```

### 模型局限性以及可能的偏差

- 该模型主要面向风格化人物生成，同时适用于场景、动物生成，但小样本数据涵盖类目有限，文本内容多样性可能有一定损失；

- 目前仅支持英文文本提示词输入；

## 训练数据介绍

- 卡通人脸数据，互联网搜集，10～100张


## 引用
如果该模型对你有所帮助，请引用相关的论文：

```BibTeX
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
