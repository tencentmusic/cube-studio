# Stable Diffusion v2 for Image Inpainting 图像填充模型
该模型为图像填充模型，输入一个抹除部分内容的图像，实现端到端的图像填充，返回填充后的完整图像。

模型效果如下:

<img src="./img/inpainting_demo.gif">

## 模型描述
该模型基于Stable Diffusion v2与diffusers进行构建。
## 模型期望使用方式和适用范围

1. 该模型适用于多种场景的图像输入，给定图像（Image）和需要修补填充区域的掩码（Mask），生成修补填充后的新图像；
2. 该模型推理时对机器GPU显存有一定要求；在FP16模式下并开启enable_attention_slicing选项时，对于16G显存的显卡，建议的最大输入分辨率为1920x1080；FP32模式建议使用含较大显存（如32G及以上）GPU的机器进行推理。如果没有GPU显卡或显存不足够，可以尝试使用CPU模式进行推理。
### 如何使用Demo Service
通过在页面右侧绘制Mask，即可快速体验模型效果：
- 建议点击右上角的最大化按钮后再绘制Mask，充分抹除物体可以带来更好的Inpainting效果；
- 希望抹除物体并还原背景时，Prompt默认为"background"；希望生成其他物体时，可以更改Prompt来描述希望生成的物体；Prompt需要英文输入；
- 建议上传体验图像的分辨率不超过1280x720，更大尺寸的图像推理时需要更大的GPU显存和更长的推理时间，可以在Notebook或本地调用Pipeline体验。
### 如何使用Pipeline
在 ModelScope 框架上，提供输入图像和掩码，即可以通过简单的 Pipeline 调用来使用Stable Diffusion v2图像填充模型。
#### 推理代码范例
```python
import cv2
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting_1.png'
input_mask_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting_mask_1.png'
prompt = 'background'
output_image_path = './result.png'

input = {
    'image': input_location,
    'mask': input_mask_location,
    'prompt': prompt
}
image_inpainting = pipeline(
    Tasks.image_inpainting,
    model='damo/cv_stable-diffusion-v2_image-inpainting_base',
    device='gpu',
    torch_dtype=torch.float16,
    enable_attention_slicing=True)
output = image_inpainting(input)[OutputKeys.OUTPUT_IMG]
cv2.imwrite(output_image_path, output)
print('pipeline: the output image path is {}'.format(output_image_path))
```
#### 推理代码说明

- Pipeline初始化参数
  - 可缺省参数device，默认值为'gpu'，可设置为'cpu'。
  - 可缺省参数torch_dtype，默认值为torch.float16，可设置为torch.float32。
  - 可缺省参数enable_attention_slicing，默认值为True，开启将减少GPU显存占用，可关闭。

- Pipeline调用参数
  - 输入要求：输入字典中必须指定的字段有'image'，'mask'；其他可选输入字段及其默认值包括：
```python
'prompt': 'background',
'num_inference_steps': 50,
'guidance_scale': 7.5,
'negative_prompt': None,
'num_images_per_prompt': 1,
'eta': 0.0
```
  - 额外参数：
    - prompt参数也支持在Pipeline调用时作为单独参数传入；但如果input中存在prompt字段，将会优先使用input中的prompt。

- 由于GPU显存限制，本项目默认支持开启FP16推理，可以在构建pipeline时传入参数torch_dtype=torch.float32来使用FP32；同时torch_dtype参数可缺省，默认值为torch.float16。
- 本项目支持使用CPU进行推理，可以在构建pipeline时传入参数device='cpu'；CPU模式下torch_dtype仅支持torch.float32。

### 模型局限性以及可能的偏差

- 实际测试中，FP16模式下生成的图像较FP32模式质量有所下降。
- 在一些背景较为简单平滑的场景下，Stable Diffusion可能生成一些无意义的前景物体，可以通过调整Prompt和模型参数进行消除。
- 在一些场景下，指定某些不同的Prompt时，Stable Diffusion可能生成错误的前景物体；可以生成多次，取效果较好的结果。
- 目前模型推理前会Resize输入图像以匹配输入尺寸要求。

## 训练介绍
本模型根据diffusers开源库构建，由Stability-AI从 stable-diffusion-2-base (512-base-ema.ckpt) 微调 200k steps。 并使用了LAMA中提出的掩码生成策略。请参考[模型来源](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)。

## 说明与引用
本算法模型源自一些开源项目：

- [https://github.com/Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)
- [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
- [https://huggingface.co/stabilityai/stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

