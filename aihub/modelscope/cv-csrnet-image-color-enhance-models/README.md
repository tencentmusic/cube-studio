
# CSRNet: Conditional Sequential Modulation for Efficient Global Image Retouching

## 模型描述
该模型为图像调色模型，输入为待调色的图像，输出为调色后的图像。CSRNet通过计算全局调整参数并将之作用于条件网络得到的特征，保证效果的基础之上实现轻便高效的训练和推理。

<img src="./data/csrnet_1.png" width=600 alt="CSRNet architecture">


## 期望模型使用方式以及适用范围

适用于一般条件下拍摄得到的图像，可提升图像的色彩质感。

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_color_enhance.png'
image_color_enhance = pipeline(Tasks.image_color_enhancement, 
                               model='damo/cv_csrnet_image-color-enhance-models')
result = image_color_enhance(img)
cv2.imwrite('enhanced_result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差
模型基于adobefivek进行训练，对于与该数据集分布相差较多的输入可能难以达到最佳效果，在某些极端情形下效果亦无法保证。

## 训练数据介绍
Adobefivek数据集，包含4500个图像对（手机拍摄的待调色图像与经艺术家调整后的图像），两组图像对来自同一个场景实例。

训练数据链接: https://data.csail.mit.edu/graphics/fivek/

## 测试数据介绍
Adobefivek数据集

文件类型： .PNG

文件数量： 500

包含500个图像对，均为来自同一个场景的实例。

## 数据评估及结果
| name | Dataset | PSNR | SSIM |
|:---- |:----    |:---- |:----|
|Adove5K(512)|Adobe5K|25.02|0.9256|

### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：

```@article{DBLP:journals/corr/abs-2009-10390,
  author    = {Jingwen He and
               Yihao Liu and
               Yu Qiao and
               Chao Dong},
  title     = {Conditional Sequential Modulation for Efficient Global Image Retouching},
  journal   = {CoRR},
  volume    = {abs/2009.10390},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.10390},
  eprinttype = {arXiv},
  eprint    = {2009.10390},
  timestamp = {Thu, 12 Nov 2020 16:03:58 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2009-10390.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

https://github.com/hejingwenhejingwen/CSRNet
