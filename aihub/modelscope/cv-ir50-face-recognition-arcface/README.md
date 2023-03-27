
# ArcFace 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFace&spm=a2cio.27993362)、[口罩人脸比对1:1](https://vision.aliyun.com/experience/detail?tagName=facebody&children=CompareFaceWithMask&spm=a2cio.27993362)、[人脸搜索1:N](https://vision.aliyun.com/experience/detail?tagName=facebody&children=SearchFace&spm=a2cio.27993362)、[公众人物识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=RecognizePublicFace&spm=a2cio.27993362)、[明星识别](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectCelebrity&spm=a2cio.27993362)。

人脸识别模型ArcFace, 推荐使用An Efficient Training Approach for Very Large Scale Face Recognition ([代码地址](https://github.com/tiandunx/FFC))框架快速训练。


## 模型描述

ArcFace为近几年人脸识别领域的代表性工作，被CVPR2019录取([论文地址](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)), [代码地址](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch))，该方法主要贡献是提出了ArcFace loss, 在$x_i$和$W_{ji}$之间的θ上加上角度间隔m（注意是加在了角θ上），以加法的方式惩罚深度特征与其相应权重之间的角度，从而同时增强了类内紧度和类间差异。由于提出的加性角度间隔(additive angular margin)惩罚与测地线距离间隔(geodesic distance margin)惩罚在归一化的超球面上相等，因此作者将该方法命名为ArcFace。此外作者在之后的几年内持续优化该算法，使其一直保持在sota性能。

## 模型结构
![模型结构](https://modelscope.cn/api/v1/models/damo/cv_ir50_face-recognition_arcface/repo?Revision=master&FilePath=arcface.jpg&View=true)

## 模型使用方式和使用范围
本模型可以检测输入图片中人脸的特征

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

arc_face_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_ir50_face-recognition_arcface')
img1 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png'
img2 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_2.png'
emb1 = arc_face_recognition_func(img1)[OutputKeys.IMG_EMBEDDING]
emb2 = arc_face_recognition_func(img2)[OutputKeys.IMG_EMBEDDING]
sim = np.dot(emb1[0], emb2[0])
print(f'Face cosine similarity={sim:.3f}, img1:{img1}  img2:{img2}')
```

### 使用方式
- 推理：输入经过对齐的人脸图片(112x112)，返回人脸特征向量(512维)，为便于体验，集成了人脸检测和关键点模型RetinaFace，输入两张图片，各自进行人脸检测选择最大脸并对齐后提取特征，然后返回相似度比分


### 目标场景
- 人脸识别应用广泛，如考勤，通行，人身核验，智慧安防等场景


### 模型局限性及可能偏差
- 训练数据仅包括ms1mv3数据集，模型鲁棒性可能有所欠缺。
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 模型性能指标

| Method | IJBC(1e-5) | IJBC(1e-4) | MFR-ALL |
| ------------ | ------------ | ------------ | ------------ |
| ArcFace | 94.07  | 95.97 | 75.13 |


## 人脸相关模型

以下是ModelScope上人脸相关模型:

- 人脸检测

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [RetinaFace人脸检测模型](https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary) |
| 2 | [MogFace人脸检测模型-large](https://modelscope.cn/models/damo/cv_resnet101_face-detection_cvpr22papermogface/summary) |
| 3 | [TinyMog人脸检测器-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_tinymog/summary) |
| 4 | [ULFD人脸检测模型-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_ulfd/summary) |
| 5 | [Mtcnn人脸检测关键点模型](https://modelscope.cn/models/damo/cv_manual_face-detection_mtcnn/summary) |
| 6 | [ULFD人脸检测模型-tiny](https://modelscope.cn/models/damo/cv_manual_face-detection_ulfd/summary) |


- 人脸识别

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [口罩人脸识别模型FaceMask](https://modelscope.cn/models/damo/cv_resnet_face-recognition_facemask/summary) |
| 2 | [口罩人脸识别模型FRFM-large](https://modelscope.cn/models/damo/cv_manual_face-recognition_frfm/summary) |
| 3 | [IR人脸识别模型FRIR](https://modelscope.cn/models/damo/cv_manual_face-recognition_frir/summary) |
| 4 | [ArcFace人脸识别模型](https://modelscope.cn/models/damo/cv_ir50_face-recognition_arcface/summary) |
| 5 | [IR人脸识别模型FRIR](https://modelscope.cn/models/damo/cv_manual_face-recognition_frir/summary) |

- 人脸活体识别

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [人脸活体检测模型-IR](https://modelscope.cn/models/damo/cv_manual_face-liveness_flir/summary) |
| 2 | [人脸活体检测模型-RGB](https://modelscope.cn/models/damo/cv_manual_face-liveness_flrgb/summary) |
| 3 | [静默人脸活体检测模型-炫彩](https://modelscope.cn/models/damo/cv_manual_face-liveness_flxc/summary) |

- 人脸关键点

| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [FLCM人脸关键点置信度模型](https://modelscope.cn/models/damo/cv_manual_facial-landmark-confidence_flcm/summary) |

- 人脸属性 & 表情


| 序号 | 模型名称 |
| ------------ | ------------ |
| 1 | [人脸表情识别模型FER](https://modelscope.cn/models/damo/cv_vgg19_facial-expression-recognition_fer/summary) |
| 2 | [人脸属性识别模型FairFace](https://modelscope.cn/models/damo/cv_resnet34_face-attribute-recognition_fairface/summary) |

## 来源说明
本模型及代码来自开源社区([地址](https://github.com/deepinsight/insightface))，请遵守相关许可。

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{deng2019arcface,
      title={Arcface: Additive angular margin loss for deep face recognition},
        author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
          booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
            pages={4690--4699},
              year={2019}
}
```

