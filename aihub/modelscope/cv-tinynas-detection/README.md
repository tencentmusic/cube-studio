
# AIRDet 介绍
AIRDet是一个面向工业落地的目标检测框架，目的是针对复杂多变的工业场景提供开箱即用的目标检测能力。

<div align="center"><img src="assets/airdet.png" width="500"></div>

## 模型描述
这里我们提供基于检测框架[AIRDet](https://github.com/tinyvision/AIRDet)训练的轻量化检测模型：AIRDet-S。AIRDet-S中使用了引入了Giraffe neck、GFLv2 head、AutoAugmentation等技术，使其在精度上超越了目前的一众YOLO(YOLOX-s, YOLOv6-s, YOLOe-s)，并且仍然保持极高的推理速度。

## 期望模型使用方式与适用范围
本模型适用范围较广，能对图片中包含的大部分前景物体（COCO 80类）进行定位。
### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用当前模型，模型将输出图片中物体的坐标。
#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
object_detect = pipeline(Tasks.image_object_detection,model='damo/cv_tinynas_detection')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'
result = object_detect(img_path)
print(result)
```
### 模型局限性以及可能的偏差
考虑 GPU 精度等的差异，可能带来一定的性能差异

## 训练数据介绍
使用标准COCO-2017 数据集。COCO-2017数据集由Microsoft团队提供，包含118287张训练图片、5000张验证图片、40670张测试图片，共包括80类物体。

## 数据评估及结果
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency V100<br>TRT-FP32-BS32| Latency V100<br>TRT-FP16-BS32| FLOPs<br>(G)| weights |
| ------        |:---: | :---:     |:---:|:---: | :---: | :----: |
|[Yolox-S](./configs/yolox_s.py)   | 640 | 40.5 | 3.4 | 2.3 | 26.81 | [link]() |
|[AIRDet-S](./configs/airdet_s.py) | 640 | 44.2 | 4.4 | 2.8 | 27.56 | [link](https://drive.google.com/file/d/119W87oZ4zcJvvjzYCmBudX38cRpZbQc4/view?usp=sharing) |


- 上表中汇报的是COCO2017 val集上的结果。
- latency以毫秒为单位，计时不包括后处理时间。

## 引用
```BibTex
@article{jiang2022giraffedet,
  title={GiraffeDet: A Heavy-Neck Paradigm for Object Detection},
    author={Jiang, Yiqi and Tan, Zhiyu and Wang, Junyan and Sun, Xiuyu and Lin, Ming and Li, Hao},
      journal={arXiv preprint arXiv:2202.04256},
        year={2022}
}
@inproceedings{li2021generalized,
  title={Generalized focal loss v2: Learning reliable localization quality estimation for dense object detection},
    author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={11632--11641},
          year={2021}
}
        
```
