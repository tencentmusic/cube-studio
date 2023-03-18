
# Paint by Example 图像示例替换

本模型选自Paint by Example， 根据示例图片对原始图片的mask区域进行自适应地替换。

<img src="teaser.png" width=70% />

## 模型描述

Paint by Example 是基于stable diffusion 模型的一个图像编辑算法，根据示例图片对原始图片的mask区域进行自适应地替换。

## 期望模型使用方式以及适用范围

本模型适用范围为室外自然场景；

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 环境安装

安装好基础modelscope环境后，安装paint-ldm：

pip install paint-ldm -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html


#### 代码范例


- 推理(仅支持GPU)：


```python
from PIL import Image
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_paint_by_example/image/example_1.png'
input_mask_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_paint_by_example/mask/example_1.png'
reference_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_paint_by_example/reference/example_1.jpg'
input = {
        'img':input_location,
        'mask':input_mask_location,
        'reference':reference_location,
}

paintbyexample = pipeline(Tasks.image_paintbyexample, model='damo/cv_stable-diffusion_paint-by-example')
result = paintbyexample(input)
vis_img = result[OutputKeys.OUTPUT_IMG]
cv2.imwrite("result.png", vis_img)
```


### 模型局限性以及可能的偏差

- 人脸图片暂不支持
- 当前版本在python 3.8环境测试通过，其他环境下可用性待测试
- 当前版本fine-tune在cpu和单机单gpu环境测试通过，单机多gpu等其他环境待测试

## 训练数据介绍

- [ImageNet](https://www.image-net.org/)：包括1M训练数据




## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@article{yang2022paint,
  title={Paint by Example: Exemplar-based Image Editing with Diffusion Models},
  author={Binxin Yang and Shuyang Gu and Bo Zhang and Ting Zhang and Xuejin Chen and Xiaoyan Sun and Dong Chen and Fang Wen},
  journal={arXiv preprint arXiv:2211.13227},
  year={2022}
}
```
