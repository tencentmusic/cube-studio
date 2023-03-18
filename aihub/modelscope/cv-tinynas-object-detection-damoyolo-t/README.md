
<div align="center"><img src="assets/damoyolo.png" width="2000"></div>

## 模型描述
这里我们提供基于业界领先的目标检测框架**DAMO-YOLO**训练的检测模型：**DAMO-YOLO-T**。DAMO-YOLO是一个面向工业落地的目标检测框架，兼顾模型速度与精度，其训练的模型效果超越了目前的一众YOLO系列方法，并且仍然保持极高的推理速度。DAMO-YOLO引入TinyNAS技术，使得用户可以根据硬件算力进行低成本的检测模型定制，提高硬件利用效率并且获得更高精度。另外，DAMO-YOLO还对检测模型中的neck、head结构设计，以及训练时的标签分配、数据增广等关键因素进行了优化，具体细节可以参考我们的[开源代码](https://github.com/tinyvision/damo-yolo)和[技术报告](https://arxiv.org/pdf/2211.15444v2.pdf)。**DAMO-YOLO-T**是DAMO-YOLO提供的一系列模型中，平衡了速度和精度的最优模型之一。

<div align="center"><img src="assets/curve-star-t.png" width="500"></div>

## 模型评测
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency(ms)<br>T4-TRT-FP16| FLOPs<br>(G)| Parameters(M)| 
| ------        |:---: | :---: |:---:|:---: | :---: |
|**YOLOX-T** | 416  | 32.8  | 1.78| 6.5 | 5.1  | 
|**YOLOv5-N** | 640  | 28.0  | 2.23| 4.5 |1.9  | 
|**YOLOv5-S** | 640  | 37.4  | 3.04| 16.5 | 7.2  | 
|**YOLOv6-T** | 640  | 40.3  | 2.53| 36.7 | 15.0  | 
|**YOLOv7-T** | 640  | 38.7  | 3.13|13.7 | 6.2  | 
|**DAMO-YOLO-T** | 640  | 43.0  | 2.78| 18.1 | 8.5  | 

- 表中汇报的mAP是COCO2017 val集上的结果。
- 表中汇报的latency不包括后处理（nms）时间，其测试条件为：T4 GPU，TensorRT=7.2.1.6， CUDA=10.2, CUDNN=8.0.0.1。
  
## 使用范围
本模型适用范围较广，能对图片中包含的大部分前景物体（COCO 80类）进行定位。
  
## 使用方法
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。具体代码示例如下：
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
object_detect = pipeline(Tasks.image_object_detection,model='damo/cv_tinynas_object-detection_damoyolo-t')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'
result = object_detect(img_path)
```
  
## 模型可视化效果
  
<div align="center"><img src="assets/comparison.png" width="750"></div>

## 引用

```latex
 @article{damoyolo,
  title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
  author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
  journal={arXiv preprint arXiv:2211.15444v2},
  year={2022}
}
```
