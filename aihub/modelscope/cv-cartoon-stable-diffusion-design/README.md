
# 卡通系列文生图模型

输入一段文本提示词，实现特定风格卡通图像生成，返回符合文本描述且满足特定风格的结果图像。

ModelScope上提供多种风格效果的卡通生成模型：

| [<img src="description/sim1.png" width="240px">](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_design/summary) | [<img src="description/sim2.png" width="240px">](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_watercolor) | [<img src="description/sim3.png" width="240px">](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_illustration/summary)| [<img src="description/sim4.png" width="240px">](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_clipart/summary)| [<img src="description/sim5.png" width="240px">](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_flat/summary)|
|:--:|:--:|:--:|:--:|:--:| 
| [插画风格](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_design/summary) | [水彩风格](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_watercolor/summary) | [漫画风格](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_illustration/summary) | [剪贴画](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_clipart/summary) | [扁平风格](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_flat/summary) | 


## 卡通系列文生图模型-插画风

本仓库提供插画风格文生图模型，其生成效果如下所示：

(1) 公众人物生成

![生成效果](description/demo.png)

(2) 物体场景生成

![生成效果](description/demo1.png)


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

pipe = pipeline(Tasks.text_to_image_synthesis, model='damo/cv_cartoon_stable_diffusion_design', model_revision='v1.0.0')
output = pipe({'text': 'sks style, a portrait painting of Johnny Depp'})
cv2.imwrite('result.png', output['output_imgs'][0])
print('Image saved to result.png')
print('finished!')

# 更佳实践
pipe = pipeline(Tasks.text_to_image_synthesis, model='damo/cv_cartoon_stable_diffusion_design', model_revision='v1.0.0')
from diffusers.schedulers import EulerAncestralDiscreteScheduler
pipe.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipeline.scheduler.config)
output = pipe({'text': 'sks style, a portrait painting of Johnny Depp'})
cv2.imwrite('result.png', output['output_imgs'][0])
print('Image saved to result.png')
print('finished!')

```

### 模型局限性以及可能的偏差

- 该模型主要面向风格化人物生成，同时适用于场景、动物生成，但小样本数据涵盖类目有限，文本内容多样性可能有一定损失；


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
