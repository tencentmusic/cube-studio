
# AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement

## 模型描述
该模型为图像色彩增强模型，输入为待调色的图像，输出为增强后的图像。Adaptive-Interval 3DLUT 基于 3DLUT 的调色模型基础之上，将3DLUT的采样改进为自适应的采样间隔，能提高 3DLUT 网格的精度，获得更好的增强效果。

<img src="./data/adaint_1.png" width=800 alt="AdaInt architecture">


## 期望模型使用方式以及适用范围

适用于一般条件下拍摄得到的图像，可对图像进行色彩增强。

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
该模型使用JIT方式编译运行CUDA extension，因此暂不支持CPU推理，请确保在GPU 环境运行。

#### 代码范例
```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_color_enhance.png'
image_color_enhance = pipeline(Tasks.image_color_enhancement, 
                               model='damo/cv_adaint_image-color-enhance-models')
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
|Adove5K(512)|Adobe5K|25.49|0.926|

### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@InProceedings{yang2022adaint,
  title={AdaInt: Learning adaptive intervals for 3d lookup tables on real-time image enhancement},
  author={Canqian Yang and Meiguang Jin and Xu Jia and Yi Xu and Ying Chen},
  booktitle=CVPR,
  year={2022}
}
```

https://github.com/imcharlesy/adaint


