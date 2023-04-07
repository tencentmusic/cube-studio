
# ControlNet可控图像生成
## News: 点击[创空间](https://www.modelscope.cn/studios/dienstag/controlnet_controllable-image-generation/summary)即可快速体验模型！
该模型为图像生成模型，输入一张图像，指定控制类别并提供期望生成图像的描述prompt，网络会根据输入图像抽取相应的控制信息并生成精美图像。

<img src="./control_demo.gif">

## 模型描述
ControlNet可以控制预训练的大型扩散模型以支持额外的输入，其以端到端的方式学习与任务相关的特定条件。其可以增强像 Stable Diffusion 这样的大型扩散模型，从而支持输入边缘图、分割图、关键点等来生成图像。
## 模型期望使用方式和适用范围
ControlNet支持输入边缘图、分割图、简笔画、人体姿态等控制信息，本项目支持选择不同的控制信息来生成相应的图像。
### 如何使用

1. 在[创空间](https://www.modelscope.cn/studios/dienstag/controlnet_controllable-image-generation/summary)中，上传图像并选择控制信息类别即可快速体验。
2. 在 ModelScope 框架上，提供输入图像、控制信息类别和文字引导prompt，即可以通过简单的 Pipeline 调用来使用ControlNet可控图像生成模型。
#### 推理代码范例
```python
import cv2
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_segmentation.jpg'
prompt = 'person'
output_image_path = './result.png'
input = {
    'image': input_location,
    'prompt': prompt
}

pipe = pipeline(
    Tasks.controllable_image_generation,
    model='dienstag/cv_controlnet_controllable-image-generation_nine-annotators')
output = pipe(input, control_type='hed')[OutputKeys.OUTPUT_IMG]
cv2.imwrite(output_image_path, output)
print('pipeline: the output image path is {}'.format(output_image_path))
```
#### 推理代码说明

- 推理参数要求：control_type可选字段包括canny，hough，hed，depth，normal，pose，seg，fake_scribble，scribble；
  - scribble控制模式要求输入黑白简笔画图像，其余控制模式输入自然图像即可；
  - hough控制模式建议输入含较多直线的图像，如建筑物图像等；
  - pose控制模式建议输入人像，尤其是全身人像，以便提升姿态估计质量；

- 输入要求：输入字典input中，'image'为必须指定的字段，'prompt'为可缺省字段，也可以在调用pipeline时作为额外参数传入prompt，也可以置空：
```python
output = scribble_to_image(input, prompt='hot air balloon')[OutputKeys.OUTPUT_IMG]
```
- input中可选的字段及其默认值还包括
```python
"image_resolution": 512,
"strength": 1.0,
"guess_mode": False,
"ddim_steps": 20,
"scale": 9.0,
"num_samples": 1,
"eta": 0.0,
"a_prompt": "best quality, extremely detailed",
"n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
```

- 出于对速度和显存占用方面的考虑，本项目默认开启了enable_sliced_attention。
- 本项目暂时仅支持使用GPU进行推理，推荐使用显存16G及以上的GPU。

### 模型局限性以及可能的偏差

- Prompt暂时仅支持英文输入。
- 所提供的图像或简笔画过于简单或意义不明确时，模型可能生成与上传图像相关度低的物体或是一些无意义的前景物体，可以修改上传图像重新尝试。
- 在一些场景下，描述Prompt不够明确时，模型可能生成错误的前景物体，可以更改Prompt并生成多次，取效果较好的结果。
- 当所提供的图像或简笔画与描述Prompt相关度低或无关时，模型可能生成偏向图像或偏向Prompt的内容，也可能生成无意义的内容；因此建议描述Prompt与所上传的图像紧密相关并且尽可能详细。

## 说明与引用
本算法模型构建过程参考了一些开源项目：

- [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- [https://huggingface.co/lllyasviel/ControlNet](https://huggingface.co/lllyasviel/ControlNet)

如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
  author={Lvmin Zhang and Maneesh Agrawala},
  year={2023},
  eprint={2302.05543},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

