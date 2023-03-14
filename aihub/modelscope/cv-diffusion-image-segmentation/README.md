
# DDPM-Seg模型介绍

本模型使用预训练好的[guided diffusion]()作为特征提取器，并在有标注的训练数据很少的情况下取得了SOTA的结果。


本模型的生成效果如下所示：



## 模型描述

本模型研究了执行反向扩散过程的扩散网络的中间激活值，并利用这些激活值作为输入图像的语义级别特征表示。
借助强大的扩散模型作为特征提取器，本模型提供了一种简单的语义分割方法，
即使只提供少量训练图像也可以取得优越的结果。
具体的模型如下图所示


<p align="center">
    <br>
    <img src="data/ddpm-seg-schema.jpg" width="800" />
    <br>
<p>

## 期望模型使用方式以及适用范围

使用方式：
- 直接推理，输入图像，对图像中的内容进行分割

适用范围：
- 本模型目前主要应用于人脸图像（FFHQ）的分割。
- 卧室（LSUN-Bedroom）、猫个体（LSUN-Cat）和马个体（LSUN-Horse）的分割模型后续会上线。



### 如何使用
在ModelScope框架上，提供输入图像，即可以通过简单的Pipeline调用来使用本模型。



#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_ffhq34_00041527.png'
pp = pipeline(Tasks.semantic_segmentation, model='damo/cv_diffusion_image-segmentation')
result = pp(input_location)

# if you want to save the result, you can run
vis_img = result[OutputKeys.OUTPUT_IMG]
vis_img.save('result.png')
```


### 模型局限性和可能的偏差
- 当前提供的模型只在FFHQ数据集进行了训练，因此非人脸域的图像可能会影响分割结果。后续会提供在LSUN-Cat、LSUN-Horse、LSUN-Bedroom、ADE-Bedroom和CelebAMask-HQ上训练的模型。
- 模型当前只支持长宽相等的输入图像。
- 当前模型不支持CPU环境。

## 训练数据介绍
[训练数据](https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/datasets.tar.gz)为从FFHQ-256、LSUN、ADE20K和CelebAMask-HQ中收集的子集。


## 模型推理流程

### 预处理
- 调整输入图像大小为256*256
- 对输入图像进行归一化


### 推理
- 使用guided diffusion进行反向扩散
- 得到UNet的中间层激活值作为输入图像的语义特征
- 将得到的语义特征输入MLP模型，并得到最终的结果

## 数据评估及结果
| **Method**        | **Bedroom\-28** | **FFHQ\-34** | **Cat\-15**  | **Horse\-21** | **CelebA\-19** | **ADE\-Bedroom\-30** |
|-------------------|-----------------|--------------|--------------|---------------|----------------|-----------------------|
| **ALAE**          | 20\.0 ± 1\.0    | 48\.1 ± 1\.3 | \-\-         | \-\-          | 49\.7 ± 0\.7   | 15\.0 ± 0\.5          |
| **VDVAE**         | \-\-            | 57\.3 ± 1\.1 | \-\-         | \-\-          | 54\.1 ± 1\.0   | \-\-                  |
| **GAN Inversion** | 13\.9 ± 0\.6    | 51\.7 ± 0\.8 | 21\.4 ± 1\.7 | 17\.7 ± 0\.4  | 51\.5 ± 2\.3   | 11\.1 ± 0\.2          |
| **GAN Encoder**   | 22\.4 ± 1\.6    | 53\.9 ± 1\.3 | 32\.0 ± 1\.8 | 26\.7 ± 0\.7  | 53\.9 ± 0\.8   | 15\.7 ± 0\.3          |
| **SwAV**          | 41\.0 ± 2\.3    | 54\.7 ± 1\.4 | 44\.1 ± 2\.1 | 51\.7 ± 0\.5  | 53\.2 ± 1\.0   | 30\.3 ± 1\.5          |
| **SwAVw2**        | 42\.4 ± 1\.7    | 56\.9 ± 1\.3 | 45\.1 ± 2\.1 | 54\.0 ± 0\.9  | 52\.4 ± 1\.3   | 30\.6 ± 1\.0          |
| **MAE**           | 45\.0 ± 2\.0    | 58\.8 ± 1\.1 | 52\.4 ± 2\.3 | 63\.4 ± 1\.4  | 57\.8 ± 0\.4   | 31\.7 ± 1\.8          |
| **DatasetGAN**    | 31\.3 ± 2\.7    | 57\.0 ± 1\.0 | 36\.5 ± 2\.3 | 45\.4 ± 1\.4  | \-\-           | \-\-                  |
| **DatasetDDPM**   | 47\.9 ± 2\.9    | 56\.0 ± 0\.9 | 47\.6 ± 1\.5 | 60\.8 ± 1\.0  | \-\-           | \-\-                  |
| **DDPM\-Seg**     | **49\.4 ± 1\.9**    | **59\.1 ± 1\.4** | **53\.7 ± 3\.3** | **65\.0 ± 0\.8**  | **59\.9 ± 1\.0**   | **34\.6 ± 1\.7**          |



### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```BibTeX
@article{baranchuk2021label,
  title={Label-efficient semantic segmentation with diffusion models},
  author={Baranchuk, Dmitry and Rubachev, Ivan and Voynov, Andrey and Khrulkov, Valentin and Babenko, Artem},
  journal={arXiv preprint arXiv:2112.03126},
  year={2021}
}
```
### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_diffusion_image-segmentation.git
```