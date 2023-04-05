
# Taiyi-Stable-Diffusion-1B-Chinese-v0.1

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)
- API：[Fengshen-OpenAPI](https://fengshenbang-lm.com/open-api)

## 简介 Brief Introduction

首个开源的中文Stable Diffusion模型，基于0.2亿筛选过的中文图文对训练。

The first open source Chinese Stable diffusion, which was trained on 20M filtered Chinese image-text pairs.


## 简介 Brief Introduction

首个开源的中英双语Stable Diffusion模型，基于0.2亿筛选过的中文图文对训练。

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 多模态 Multimodal | 太乙 Taiyi | Stable Diffusion |    1B    |     Chinese     |

## 模型信息 Model Information

我们将[Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://zero.so.com/)数据集(23M)用作预训练的数据集，先用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)对这两个数据集的图文对相似性进行打分，取CLIP Score大于0.2的图文对作为我们的训练集。 我们使用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)作为初始化的text encoder，冻住[stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)([论文](https://arxiv.org/abs/2112.10752))模型的其他部分，只训练text encoder，以便保留原始模型的生成能力且实现中文概念的对齐。该模型目前在0.2亿图文对上训练了一个epoch。 我们在 32 x A100 训练了大约100小时。该版本只是一个初步的版本，我们将持续优化并开源后续模型，欢迎交流。

We use [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/)(100M) 和 [Zero](https://zero.so.com/)(23M) as our dataset, and take the image and text pairs with CLIP Score (based on [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)) greater than 0.2 as our Training set. We use [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) as our init text encoder. To keep the powerful generative capability of stable diffusion and align Chinese concepts with the images, We only train the text encoder and freeze other part of the [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)([paper](https://arxiv.org/abs/2112.10752)) model. It takes 100 hours to train this model based on 32 x A100. This model is a preliminary version and we will update this model continuously and open sourse. Welcome to exchange！

### Result
Basic Prompt

|  铁马冰河入梦来，3D绘画。   |  飞流直下三千尺，油画。 | 女孩背影，日落，唯美插画。  |
|  ----  | ----  | ----  |
| ![](result_examples/tiema.png)  | ![](result_examples/feiliu.png)  | ![](result_examples/nvhai.jpg) |

Advanced Prompt

| 铁马冰河入梦来，概念画，科幻，玄幻，3D  | 中国海边城市，科幻，未来感，唯美，插画。 | 那人却在灯火阑珊处，色彩艳丽，古风，资深插画师作品，桌面高清壁纸。 |
|  ----  | ----  | ----  |
| ![](result_examples/tiema2.jpg)  | ![](result_examples/chengshi.jpg) | ![](result_examples/naren.jpg) |


## 使用 Usage

### 全精度 Full precision

```py
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='Fengshenbang/Taiyi-Stable-Diffusion-1B-Chinese-v0.1',
                model_revision='v1.0.0')

prompt = '飞流直下三千尺，油画'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])

```


### 半精度 Half precision FP16 (CUDA)

添加 `torch_dtype=torch.float16` 和 `device_map="auto"` 可以快速加载 FP16 的权重，以加快推理速度。
更多信息见 [the optimization docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16#half-precision-weights)。

```py
# !pip install git+https://github.com/huggingface/accelerate
import torch
from diffusers import StableDiffusionPipeline
torch.backends.cudnn.benchmark = True
pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", torch_dtype=torch.float16)
pipe.to('cuda')

prompt = '飞流直下三千尺，油画'
image = pipe(prompt, guidance_scale=7.5).images[0]  
image.save("飞流.png")
```

### 使用手册 Handbook for Taiyi

https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/stable_diffusion_chinese/taiyi_handbook.md

### 怎样微调 How to finetune

https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/finetune_taiyi_stable_diffusion

### webui配置 Configure webui

https://github.com/IDEA-CCNL/stable-diffusion-webui/blob/master/README.md

### DreamBooth

https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/stable_diffusion_dreambooth

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[总论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen and Ruyi Gan and Jiaxing Zhang},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```

