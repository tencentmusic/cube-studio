
# TAdaConv 模型介绍


## 模型描述

TAdaConv是一种在行为识别模型中即插即用的时序自适应卷积（Temporally-Adaptive Convolutions）。作为2D/3D卷积的增强版，TAdaConv可以明显提升SlowFast、R2D和R3D等任何使用2D/3D卷积的模型性能，而额外的计算量可以忽略不计。在具体实现过程中，TAdaConv将每一帧的卷积参数分解为一个基础权重（base weight）和一个校准权重（calibration weight）。其中，基础权重 [w_b] 对于所有帧相同，而校准权重 [a_t] 则对于每一帧独立。为度量TAdaConv性能，分别在行为识别、时序行为检测上进行验证，公有数据集包括：Kinetics400、Something-Something-2和EPIC-Kitchens-100。

其模型结构如下所示：

![模型结构](https://modelscope.cn/api/v1/models/damo/cv_TAdaConv_action-recognition/repo?Revision=master&FilePath=description/model_image.jpg&View=true)


## 使用方式和范围

使用方式：
- 直接推理，在K400、SSV2、EK100支持的标签集上进行直接推理;
- 微调，在已经公开的模型在新数据、新任务上进行微调

使用范围:
- 适合视频领域的行为识别检测识别，分辨率在224x224以上，输入片段宜在10s以内

目标场景:
- 视频中的动作识别，比如体育、影视、直播等


### 如何使用

TAdaConv是即插即用的模型，可以直接嵌入到目前已知的3D网络结构当中。


#### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

recognition_pipeline = pipeline(Tasks.action_recognition, 'damo/cv_TAdaConv_action-recognition')
result = recognition_pipeline('data/test/videos/action_recognition_test_video.mp4')

print(f'recognition output: {result}.')
```

### 模型局限性以及可能的偏差

- 考虑GPU精度、视频解码工具的差异，可能带来一定的性能差异(<0.5%)
- 在CPU测试速度比V100 GPU测试速度相差约3倍
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试 


## 训练数据介绍

- [Kinetics-400](https://www.deepmind.com/open-source/kinetics) 常用行为识别的公开数据集，包含400类，总共有接近30万视频；

- [SSV2](https://developer.qualcomm.com/software/ai-datasets/something-something) 包含22w视频共计174类；

- [Epic-Kitchens-100](https://epic-kitchens.github.io/2022) 包含9万视频片段，类别有动名词组成，其中共计97个动词和300个名词

## 模型训练流程

- 在Kinetics-400上，backbone基本学习率设置为0.0001，head基本学习率为0.001，使用Adamw作为优化器。模型训练100epoch，在前8个epoch中，采用线性warmup策略，从学习率0.01开始，weight decay为 1e-4，Drop-path rate设置为0.4;

- 在Epic-Kitchens-100上，使用在Kinetics-400上预训练的权重进行初始化，训练长度减少到50
epochs，前10个epoch进行warm-up，基础学习率为 0.48;

- Something-Something-V2, 使用ImageNet预训练Resnet权重和Kinetics-400预训练的TAdaConvNeXt的权重初始化模型, 训练64epochs，前4个epoch为warm-up，采用SGD优化器，基础学习率为0.48.

### 预处理

主要是用的预处理如下：
- Temporal Jittering
- Random crop: 最小边Resize到[256, 320]，然后随机Crop 224x224


## 数据评估及结果

TAdaConv在行为识别和时序行为检测两个任务上进行测试:

- 行为识别,在共有行为识别数据集Kinetics400和Something-Something-2上的结果如下：

| Dataset | architecture | depth | #frames | acc@1 | acc@5 | 
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| SSV2 | TAda2D | R50 | 8 | 64.0 | 88.0 | 
| SSV2 | TAda2D | R50 | 16 | 65.6 | 89.1 | 
| K400 | TAda2D | R50 | 8 x 8 | 76.7 | 92.6 | 
| K400 | TAda2D | R50 | 16 x 5 | 77.4 | 93.1 |

- 行为检测,在时序行为检测数据集Epic-Kitchens-100上结果如下：

| Dataset | architecture | depth | acc@1 | 
| ------------ | ------------ | ------------ | ------------ | 
| EK100 | TSN    | R50 | 28.6 | 
| EK100 | TAda2D | R50 | 32.3 | 



## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{huang2021tada,
  title={TAda! Temporally-Adaptive Convolutions for Video Understanding},
  author={Huang, Ziyuan and Zhang, Shiwei and Pan, Liang and Qing, Zhiwu and Tang, Mingqian and Liu, Ziwei and Ang Jr, Marcelo H},
  booktitle={{ICLR}},
  year={2022}
}
```
```BibTeX
@article{qing2021stronger,
  title={A Stronger Baseline for Ego-Centric Action Detection},
  author={Qing, Zhiwu and Huang, Ziyuan and Wang, Xiang and Feng, Yutong and Zhang, Shiwei and Jiang, Jianwen and Tang, Mingqian and Gao, Changxin and Ang Jr, Marcelo H and Sang, Nong},
  journal={arXiv preprint arXiv:2106.06942},
  year={2021}
}
```
