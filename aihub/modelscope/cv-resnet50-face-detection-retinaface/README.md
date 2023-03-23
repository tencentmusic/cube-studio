
# RetinaFace 模型介绍
稳定调用及效果更好的API，详见视觉开放智能平台：[人脸检测与五官定位](https://vision.aliyun.com/experience/detail?tagName=facebody&children=DetectFace&spm=a2cio.27993362)。

人脸检测关键点模型RetinaFace


## 模型描述

RetinaFace为当前学术界和工业界精度较高的人脸检测和人脸关键点定位二合一的方法，被CVPR 2020 录取([论文地址](https://arxiv.org/abs/1905.00641), [代码地址](https://github.com/biubug6/Pytorch_Retinaface)))，该方法的主要贡献是:
- 引入关键点分支，可以在训练阶段引入关键点预测分支进行多任务学习，提供额外的互补特征，inference去掉关键点分支即可，并不会引入额外的计算量。


RetinaFace 的结构如下：

![模型结构](https://modelscope.cn/api/v1/models/damo/cv_resnet50_face-detection_retinaface/repo?Revision=master&FilePath=RetinaFace.jpg&View=true)


## 模型效果
![RetinaFace模型效果](result.png)

## 模型使用方式和使用范围
本模型可以检测输入图片中人脸以及对应关键点的位置。

### 使用方式
- 推理：输入图片，如存在人脸则返回人脸和人脸五点关键点位置，可检测多张人脸

### 目标场景
- 人脸相关的基础能力，可应用于人像美颜/互动娱乐/人脸比对等场景

### 模型局限性及可能偏差
- 小脸的检测效果一般
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试

### 预处理
测试时主要的预处理如下：
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

retina_face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/retina_face_detection.jpg'
result = retina_face_detection(img_path)
print(f'face detection output: {result}.')
```

### 模型训练流程
- 在Wider Face数据集上使用SGD优化器，Warmup 5个epoch, 初始学习率为1e-2，batch size 为32，训练了80个epoch。

### 测试集
- WIDERFACE: 测试集已上传至ModelScope的DatasetHub, 详情请见[WIDER_FACE](https://modelscope.cn/datasets/shaoxuan/WIDER_FACE)。

## 数据评估及结果
模型在WiderFace的验证集上客观指标如下：
| Method | Easy | Medium | Hard |
| ------------ | ------------ | ------------ | ------------ |
| RetinaFace | 94.8 | 93.8 | 89.6 |


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
本模型及代码来自开源社区([地址](https://github.com/biubug6/Pytorch_Retinaface))，请遵守相关许可。

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{deng2020retinaface,
      title={Retinaface: Single-shot multi-level face localisation in the wild},
        author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
          booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
            pages={5203--5212},
              year={2020}
}
```

