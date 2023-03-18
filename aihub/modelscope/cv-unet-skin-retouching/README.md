
# 基于混合图层的高清人像美肤模型

### [论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Lei_ABPN_Adaptive_Blend_Pyramid_Network_for_Real-Time_Local_Retouching_of_CVPR_2022_paper.pdf) ｜ [github](https://github.com/youngLBW/CRHD-3K)

人像美肤模型可用于对图像中的人体皮肤进行处理，实现匀肤（处理痘印、肤色不均等）、去瑕疵（脂肪粒、斑点、痣等）以及美白等功能。模型仅对裸露的皮肤进行修饰美化，不影响其他区域。

![内容图像](images/examples.jpg)

## 修图系列模型

| [<img src="images/skin_retouching_examples_3.jpg" height="200px">](https://modelscope.cn/models/damo/cv_unet_skin-retouching/summary) | [<img src="images/body_reshaping_results.jpg" height="200px">](https://modelscope.cn/models/damo/cv_flow-based-body-reshaping_damo/summary) |
|:--:|:--:| 
| [ABPN人像美肤](https://modelscope.cn/models/damo/cv_unet_skin-retouching/summary) | [FBBR人体美型](https://modelscope.cn/models/damo/cv_flow-based-body-reshaping_damo/summary) |


## 模型描述

为实现精细化的人像美肤，我们整体采用了先定位、后编辑的二阶段处理方法，且针对美肤任务中的不同瑕疵类型设计了不同的网络结构。

- 匀肤：对于匀肤这类需要处理大面积区域的任务，我们借鉴数字图像处理领域中的混合图层(blend layer)概念，基于unet设计了一个混合图层预测网络以实现目标区域的编辑。
- 去瑕疵：对于脂肪粒、痣这类局部区域的瑕疵，我们首先利用unet对于目标区域进行分割定位，而后使用inpainting网络对目标区域进行修复。
- 美白：我们利用皮肤分割算法结合混合图层的处理方式，实现皮肤区域的美白。

我们将匀肤模型中的blend layer概念进行拓展，提出基于自适应混合图层的局部修饰网络[ABPN](https://openaccess.thecvf.com/content/CVPR2022/papers/Lei_ABPN_Adaptive_Blend_Pyramid_Network_for_Real-Time_Local_Retouching_of_CVPR_2022_paper.pdf) （如下图） ，实现了端到端的局部修饰（美肤、服饰去皱等），但考虑到输入图像的分辨率、人像占比以及不同瑕疵的分布差异等问题，这里我们采用了多模型的方法以实现更精准、更鲁棒的美肤效果。

![内容图像](images/ABPN_framework.jpg)

## 期望模型使用方式以及适用范围

使用方式：
- 直接推理，输入图像直接进行推理。

使用范围:
- 适用于包含人脸的人像照片，其中人脸分辨率大于100x100，图像整体分辨率小于5000x5000。

目标场景:
- 需要进行皮肤美化的场景，如摄影修图、图像直播等。

### 如何使用

本模型基于pytorch（匀肤、去瑕疵）、tensorflow（皮肤分割）进行训练和推理，在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来使用人像美肤模型。

#### 代码范例
```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

skin_retouching = pipeline(Tasks.skin_retouching,model='damo/cv_unet_skin-retouching')
result = skin_retouching('https://modelscope.oss-cn-beijing.aliyuncs.com/demo/skin-retouching/skin_retouching_examples_1.jpg')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差
- 模型训练数据有限，部分非常规图像或者人像占比过小可能会影响皮肤分割（美白）效果。
- 在人脸分辨率大于100×100的图像上可取得期望效果，分辨率过小时皮肤区域本身比较模糊，美肤效果不明显。

## 训练数据介绍
- 对公开人脸人体数据集（FFHQ等）、互联网搜集的人像图像等进行标注，构造成对图像作为训练数据。
- 将不同类型瑕疵融合到人脸、皮肤区域，伪造训练数据。

### 预处理
- 人脸区域裁剪、resize到512x512分辨率作为匀肤、去瑕疵网络输入。
- 人体区域裁剪、resize到512x512分辨率作为皮肤分割网络输入。

### 后处理
- 将网络处理后的人脸、人体区域贴回到原图中。

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{lei2022abpn,
  title={ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo},
  author={Lei, Biwen and Guo, Xiefan and Yang, Hongyu and Cui, Miaomiao and Xie, Xuansong and Huang, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2108--2117},
  year={2022}
}
```
