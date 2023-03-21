
# DCT-Net人像卡通化-扩散模型-插画风

### [论文](https://arxiv.org/abs/2207.02426) ｜ [项目主页](https://menyifang.github.io/projects/DCTNet/DCTNet.html)

输入一张人物图像，实现端到端全图卡通化转换，生成插画风格虚拟形象，返回风格化后的结果图像。

其生成效果如下所示：

![生成效果](description/demo.gif)


## 模型描述

该任务采用一种全新的域校准图像翻译模型DCT-Net（Domain-Calibrated Translation），结合Stable-Diffusion扩散模型生成小样本的风格数据，即可训练得到高保真、强鲁棒、易拓展的人像风格转换模型，并通过端到端推理快速得到风格转换结果。
![网络结构](description/network.png)

## 使用方式和范围

使用方式：
- 支持GPU/CPU推理，在任意真实人物图像上进行直接推理;

使用范围:
- 包含人脸的人像照片（3通道RGB图像，支持PNG、JPG、JPEG格式），人脸分辨率大于100x100，总体图像分辨率小于3000×3000，低质人脸图像建议预先人脸增强处理。

目标场景:
- 艺术创作、社交娱乐、隐私保护场景，自动化生成卡通肖像。

### 如何使用

在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来使用人像卡通化模型。


#### 代码范例

- 模型推理(支持CPU/GPU)：
```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                       model='damo/cv_unet_person-image-cartoon-sd-design_compound-models', model_revision='v1.0.0')
# 图像本地路径
#img_path = 'input.png'
# 图像url链接
img_path = 'https://invi-label.oss-cn-shanghai.aliyuncs.com/label/cartoon/image_cartoon.png'
result = img_cartoon(img_path)

cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
print('finished!')

```

- 模型训练：

环境要求：tf1.14/15及兼容cuda，支持GPU训练

```python
import os
import unittest
import cv2
from modelscope.exporters.cv import CartoonTranslationExporter
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.trainers.cv import CartoonTranslationTrainer
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

model_id = 'damo/cv_unet_person-image-cartoon-sd-design_compound-models'
data_dir = MsDataset.load(
            'dctnet_train_clipart_mini_ms',
            namespace='menyifang',
            split='train').config_kwargs['split_config']['train']

data_photo = os.path.join(data_dir, 'face_photo')
data_cartoon = os.path.join(data_dir, 'face_cartoon')
work_dir = 'exp_localtoon'
max_steps = 10
trainer = CartoonTranslationTrainer(
            model=model_id,
            work_dir=work_dir,
            photo=data_photo,
            cartoon=data_cartoon,
            max_steps=max_steps)
trainer.train()
```

上述训练代码仅仅提供简单训练的范例，对大规模自定义数据，替换data_photo为真实人脸数据路径，data_cartoon为卡通风格人脸数据路径，max_steps建议设置为300000，可视化结果将存储在work_dir下；此外configuration.json(~/.cache/modelscope/hub/damo/cv_unet_person-image-cartoon_compound-models/)可以进行自定义修改；

Note: notebook预装环境下存在numpy依赖冲突，可手动更新解决：pip install numpy==1.18.5


- 卡通人脸数据获取

卡通人脸数据可由设计师设计/网络收集得到，在此提供一种基于[Stable-Diffusion风格预训练模型](https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_design/summary)的卡通数据生成方式

```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipe = pipeline(Tasks.text_to_image_synthesis, model='damo/cv_cartoon_stable_diffusion_clipart', model_revision='v1.0.0')
from diffusers.schedulers import EulerAncestralDiscreteScheduler
pipe.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipeline.scheduler.config)
output = pipe({'text': 'archer style, a portrait painting of Johnny Depp'})
cv2.imwrite('result.png', output['output_imgs'][0])
print('Image saved to result.png')

print('finished!')
```
可通过替换Johnny Depp为其他名人姓名，产生多样化风格数据，通过人脸对齐裁剪即可得到卡通人脸数据；可以通过修改pipeline的model参数指定不同风格的SD预训练模型。


### 模型局限性以及可能的偏差

- 低质/低分辨率人脸图像由于本身内容信息丢失严重，无法得到理想转换效果，可预先采用人脸增强模型预处理图像解决；

- 小样本数据涵盖场景有限，人脸暗光、阴影干扰可能会影响生成效果。

## 训练数据介绍

训练数据从公开数据集（COCO等）、互联网搜索人像图像，并进行标注作为训练数据。

- 真实人脸数据[FFHQ](https://github.com/NVlabs/ffhq-dataset)常用的人脸公开数据集，包含7w人脸图像；

- 卡通人脸数据，互联网搜集，100+张

## 模型推理流程

### 预处理

- 人脸关键点检测
- 人脸提取&对齐，得到256x256大小的对齐人脸

### 推理

- 为控制推理效率，人脸及背景resize到指定大小分别推理，再背景融合得到最终效果；
- 亦可将整图依据人脸尺度整体缩放到合适尺寸，直接单次推理

## 数据评估及结果

使用CelebA公开人脸数据集进行评测，在FID/ID/用户偏好等指标上均达SOTA结果：

| Method | FID | ID | Pref.A | Pref.B | 
| ------------ | ------------ | ------------ | ------------ | ------------ |
| CycleGAN | 57.08 | 0.55 | 7.1 | 1.4 | 
| U-GAT-IT | 68.40 | 0.58 | 5.0 | 1.5 | 
| Toonify | 55.27 | 0.62 | 3.7 | 4.2 | 
| pSp | 69.38 | 0.60 | 1.6 | 2.5 |
| Ours | **35.92** | **0.71** | **82.6** | **90.5** |


## 引用
如果该模型对你有所帮助，请引用相关的论文：

```BibTeX
@inproceedings{men2022domain,
  title={DCT-Net: Domain-Calibrated Translation for Portrait Stylization},
  author={Men, Yifang and Yao, Yuan and Cui, Miaomiao and Lian, Zhouhui and Xie, Xuansong},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  pages={1--9},
  year={2022}
}
```