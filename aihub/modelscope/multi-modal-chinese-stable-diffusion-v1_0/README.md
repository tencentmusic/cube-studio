
# 中文StableDiffusion-文本生成图像-通用领域

中文Stable Diffusion文生图模型, 输入描述文本，返回符合文本描述的2D图像。

<img src="gen_images/csd_demo.jpg"
     alt="More samples."
     style="width: 1000px;" />

## 模型描述

本模型采用的是[Stable Diffusion 2.1模型框架](https://github.com/Stability-AI/stablediffusion)，将原始英文领域的[OpenCLIP-ViT/H](https://github.com/mlfoundations/open_clip)文本编码器替换为中文CLIP文本编码器[chinese-clip-vit-huge-patch14](https://github.com/OFA-Sys/Chinese-CLIP)，并使用大规模中文图文pair数据进行训练。训练过程中，固定中文CLIP文本编码器，利用原始Stable Diffusion 2.1 权重对UNet网络参数进行初始化、利用64卡A100共训练35W steps。

## 期望模型使用方式以及适用范围

本模型适用范围较广，能基于任意中文文本描述进行推理，生成图像。

### 如何使用
在ModelScope框架上，提供输入文本，即可以通过简单的Pipeline调用来使用文本到图像生成模型。

```python
import torch
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.text_to_image_synthesis
model_id = 'damo/multi-modal_chinese_stable_diffusion_v1.0'
# 基础调用
pipe = pipeline(task=task, model=model_id)
output = pipe({'text': '中国山水画'})
cv2.imwrite('result.png', output['output_imgs'][0])
# 输出为opencv numpy格式，转为PIL.Image
# from PIL import Image
# img = output['output_imgs'][0]
# img = Image.fromarray(img[:,:,::-1])
# img.save('result.png')

# 更多参数
pipe = pipeline(task=task, model=model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
output = pipe({'text': '中国山水画', 'num_inference_steps': 50, 'guidance_scale': 7.5, 'negative_prompt':'模糊的'})
cv2.imwrite('result.png', output['output_imgs'][0])

# 采用DPMSolver
from diffusers.schedulers import DPMSolverMultistepScheduler
pipe = pipeline(task=task, model=model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
pipe.pipeline.scheduler.config)
output = pipe({'text': '中国山水画', 'num_inference_steps': 25})
cv2.imwrite('result.png', output['output_imgs'][0])

```


### 模型局限性以及可能的偏差

* 模型基于公开数据集及互联网数据进行训练，生成结果可能会存在与训练数据分布相关的偏差。
* 该模型无法实现完美的照片级生成。
* 该模型无法生成清晰的文本。
* 该模型在复杂的组合性生成任务上表现有待提升。

### 滥用、恶意使用和超出范围的使用
* 该模型未经过训练以真实地表示人或事件，因此使用该模型生成此类内容超出了该模型的能力范围。
* 禁止用于对人或其环境、文化、宗教等产生贬低、或有害的内容生成。
* 禁止用于涉黄、暴力和血腥内容生成。
* 禁止用于错误和虚假信息生成。

## 训练数据介绍

训练数据包括经中文翻译的公开数据集（LAION-400M、cc12m、Open Images）、以及互联网搜集数据，经过美学得分、图文相关性等预处理进行图像过滤，共计约4亿图文对。


## 相关论文以及引用信息

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
```BibTeX
@article{chinese-clip,
  title={Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese},
  author={Yang, An and Pan, Junshu and Lin, Junyang and Men, Rui and Zhang, Yichang and Zhou, Jingren and Zhou, Chang},
  journal={arXiv preprint arXiv:2211.01335},
  year={2022}
}
```