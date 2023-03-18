# 视频全景分割
给定一个输入视频，输出视频每一帧的全景分割掩膜，类别，分数（虚拟分数），矩形框和跟踪的id。

与图像全景分割不同之处在于视频全景分割能够得到每个物体跟踪的id。

全景分割是要分割出图像中的stuff，things。stuff是指天空，草地等不规则区域，things是指可数的物体，例如人，车，猫等。

模型暂不支持CPU，请使用GPU的实例运行。

![视频全景分割](resources/example.png)

## 模型描述
![模型结构](resources/framework.png)

如上图所示，模型包含backbone，neck和 KernelUpdateHeads三个部分。

## 期望模型使用方式与适用范围
本模型适用范围较广，能对图片中包含的大部分感兴趣物体（VIPSeg DataSet things 58类，stuff 66类）进行分割。

### 如何使用
在ModelScope框架上，提供输入视频，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
# 模型暂不支持CPU，请使用GPU的实例运行
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_swinb_video-panoptic-segmentation_vipseg'
input_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/kitti-step_testing_image_02_0000.mp4'
seg_pipeline = pipeline(Tasks.video_panoptic_segmentation, model=model_id)
result = seg_pipeline(input_url)
```

### 模型局限性以及可能的偏差
- 当前模型在VIPSeg DataSet数据训练，其他差异较大的场景可能出现精度下降
- 部分非常规图片或感兴趣物体占比太小或遮挡严重可能会影响分割结果
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试
## 训练数据介绍
- [VIPSeg DataSet](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) ：VIPSeg数据集共包含3536个视频，84750帧。并且均匀分布232个现实复杂野外场景。每个视频长度在3s～10s不等，帧间隔为 5fps。

### 预处理
测试时主要的预处理如下：
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

## 数据评估及结果
| Backbone |  Pretrain   | VPQ  | SRQ  |
|:--------:|:-----------:|:----:|:----:|
|  swinb   | COCO | 39.8 | 46.3 |

## 引用
```BibTeX
@inproceedings{li2022video,
  title={Video k-net: A simple, strong, and unified baseline for video segmentation},
  author={Li, Xiangtai and Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Cheng, Guangliang and Tong, Yunhai and Loy, Chen Change},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18847--18857},
  year={2022}
}
```

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_swinb_video-panoptic-segmentation_vipseg.git
```
