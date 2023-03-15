# hrnet-crowd-counting模型介绍
<img src=resources/crowd_counting.jpg width=25% /><img src=resources/result.jpg width=25% />人数: 148

给定一张输入图像，输出图像中人群个数的总值，以及对应的heatmap图。

模型基本原理（如下图所示）：

<img src=net_arch.png width=100% />

针对不同的domain数据，计算一个类别中心向量来表示domain的信息，随后采用这个domain-specific信息来指引网络对来自不同domain的输入图片进行学习和推理。

## 期望模型使用方式与适用范围
本模型适用范围较广，覆盖室外监控等大部分场景。
### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.utils.cv.image_utils import numpy_to_cv2img
import cv2

crowd_counting = pipeline(Tasks.crowd_counting,model='damo/cv_hrnet_crowd-counting_dcanet')
results = crowd_counting('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/crowd_counting.jpg')
print('scores:', results[OutputKeys.SCORES])
vis_img = results[OutputKeys.OUTPUT_IMG]
vis_img = numpy_to_cv2img(vis_img)
cv2.imwrite('result.jpg', vis_img)
```
### 模型局限性以及可能的偏差
- 在室内野外区域可能存在误检
## 模型训练介绍
- 采用ShanghaiTech-A/B, QNRF数据集联合训练完成
## 模型训练流程
- 模型训练
  以HRNet作为backbone，参考 https://github.com/Zhaoyi-Yan/DCANet 训练流程
### 图片预处理
- 图像resize，长边最大2048，短边最小416，同时保持长宽比；该操作已在内部集成，用户直接输入图片即可
## 数据评估及结果
SHA/SHB/QNRF的结果分别是58.77, 7.06, 88.35，我们提供了[SHA](https://modelscope.cn/datasets/modelscope/ShanghaiTech-A/summary)，[SHB](https://modelscope.cn/datasets/modelscope/ShanghaiTech-B/summary)的评估数据和相关代码可供复现。

SHA地址：https://modelscope.cn/datasets/modelscope/ShanghaiTech-A/summary

SHB地址：https://modelscope.cn/datasets/modelscope/ShanghaiTech-B/summary

## 引用
```BibTeX
@ARTICLE{yan2021towards,
  author={Yan, Zhaoyi and Li, Pengyu and Wang, Biao and Ren, Dongwei and Zuo, Wangmeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Towards Learning Multi-domain Crowd Counting}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2021.3137593}}
```
