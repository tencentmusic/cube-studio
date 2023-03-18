
<div align="center"><img src="assets/damoyolo.png" width="2000"></div>

## 模型描述
这里我们提供基于业界领先的目标检测框架**DAMO-YOLO**训练的检测模型：**DAMO-YOLO-S**。DAMO-YOLO是一个面向工业落地的目标检测框架，兼顾模型速度与精度，其训练的模型效果超越了目前的一众YOLO系列方法，并且仍然保持极高的推理速度。DAMO-YOLO引入TinyNAS技术，使得用户可以根据硬件算力进行低成本的检测模型定制，提高硬件利用效率并且获得更高精度。另外，DAMO-YOLO还对检测模型中的neck、head结构设计，以及训练时的标签分配、数据增广等关键因素进行了优化，具体细节可以参考我们的[开源代码](https://github.com/tinyvision/damo-yolo)和[技术报告](https://arxiv.org/pdf/2211.15444v2.pdf)。**DAMO-YOLO-S**是DAMO-YOLO提供的一系列模型中，平衡了速度和精度的最优模型之一。为了方便用户使用DAMO-YOLO完成检测任务，我们开源了多个[工业应用模型](#工业应用模型)，欢迎试用。

<div align="center"><img src="assets/curve-star-s.png" width="500"></div>

## 模型评测
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency(ms)<br>T4-TRT-FP16| FLOPs<br>(G)| Parameters(M)| 
| ------        |:---: | :---: |:---:|:---: | :---: |
|YOLOX-S        | 640  | 40.5  | 3.20| 26.8 | 9.0   |         
|YOLOv5-S       | 640  | 37.4  | 3.04| 16.5 | 7.2   |         
|YOLOv6-S       | 640  | 43.5  | 3.10| 44.2 | 17.0  | 
|PP-YOLOE-S     | 640  | 43.0  | 3.21| 17.4 | 7.9   | 
|**DAMO-YOLO-S** | 640  | 46.8  | 3.83| 37.8 | 16.3  | 

- 表中汇报的mAP是COCO2017 val集上的结果。
- 表中汇报的latency不包括后处理（nms）时间，其测试条件为：T4 GPU，TensorRT=7.2.1.6， CUDA=10.2, CUDNN=8.0.0.1。
  
## 使用范围
本模型适用范围较广，能对图片中包含的大部分前景物体（COCO 80类）进行定位。
  
## 使用方法
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。具体代码示例如下：
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
object_detect = pipeline(Tasks.image_object_detection,model='damo/cv_tinynas_object-detection_damoyolo')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'
result = object_detect(img_path)
```

## 训练示例
DAMO-YOLO现已支持使用自定义数据训练，欢迎试用！如在使用中发现问题，欢迎反馈给xianzhe.xxz@alibaba-inc.com。

在ModelScope上使用自定义数据训练DAMO-YOLO有三个关键步骤，一个简单示例如下：

**步骤一**：将标签数据转换成[COCO](https://cocodataset.org/)格式，COCO格式下一条有效数据的示例如下：
```json
{
  "categories": 
  [{
      "supercategory": "person", 
      "id": 1, 
      "name": "person"
  }], 
 "images": 
  [{
      "license": 1, 
      "file_name": "000000425226.jpg",        
      "coco_url": "http://images.cocodataset.org/val2017/000000425226.jpg", 
      "height": 640, 
      "width": 480, 
      "date_captured": 
      "2013-11-14 21:48:51", 
      "flickr_url": 
      "http://farm5.staticflickr.com/4055/4546463824_bc40e0752b_z.jpg", 
      "id": 1
  }], 
 "annotations": 
  [{
      "image_id": 1, 
      "category_id": 1, 
      "segmentation": [], 
      "area": 47803.279549999985, 
      "iscrowd": 0, 
      "bbox": [73.35, 206.02, 300.58, 372.5], 
      "id": 1
  }]
}
```
随后，可以将您的自定数据组织成如下结构：
```
├── custom_data
│   ├── annotations
│   │   └── toy_sample.json
│   ├── images
│   │   └── 000000425226.jpg
```

**步骤二**：使用默认配置文件进行训练或者自定义配置文件进行训练，训练结果将保存在./workdirs下。

+ **使用默认配置文件进行训练**：
此时，trainer会自动从‘damo/cv_tinynas_object-detection_damoyolo’下载模型训练所需的配置文件，并使用默认配置进行训练。

```python
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

cache_path = "./visdrone"
kwargs = dict(
            cfg_file=os.path.join(cache_path, 'damoyolo_tinynasL25_S_visdrone.py'),
            gpu_ids=[  # 指定训练使用的gpu
                0,1,2,3,4,5,6,7
            ],
            num_classes=10, # 自定义数据中的类别数
            train_image_dir='./data/visdrone/VisDrone2019-DET-train/images', # 训练图片路径
            val_image_dir='./data/visdrone/VisDrone2019-DET-val/images', # 测试图片路径
            train_ann=
            './data/visdrone/VisDrone2019-DET-train/annotations/visdrone_train.json', # 训练标注文件路径
            val_ann=
            './data/visdrone/VisDrone2019-DET-val/annotations/visdrone_val.json', # 测试标注文件路径
            )
trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
trainer.train() # 训练log将会保存在./workdirs/damoyolo_s/train_log.txt

```

+ **使用自定义配置文件进行训练**：
假设您有自定义的配置文件以及模型训练必须的骨干网络结构、预训练权重等文件位于"./custom"，想要使用您的自定义配置文件进行训练，可以参考如下示例：
```python
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

kwargs = dict(
            cfg_file=os.path.join(cache_path, 'configuration.json'),
            gpu_ids=[
                0,
            ],
            batch_size=2,
            max_epochs=3,
            num_classes=80,
            load_pretrain=True,
            pretrain_model='pretrain_weight.pth' # 指定预训练模型，该预训练模型需要放置在cache_path目录下，
                                                 # 只有load_pretrain=True，该配置才生效。
            base_lr_per_img=0.001,
            cache_path=cache_path,
            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',
            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',
        )
        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()
        trainer.evaluate(
            checkpoint_path=os.path.join(cache_path,
                                         'damoyolo_tinynasL25_S.pt')) # 验证模型精度


```



## 工业应用模型
我们提供了一系列面向实际工业场景的DAMO-YOLO模型，欢迎试用。请保持持续关注，更多的重磅模型即将释出！

|[**人体检测**](https://www.modelscope.cn/models/damo/cv_tinynas_human-detection_damoyolo/summary)| [**安全帽检测**](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_safety-helmet/summary)|
| :---: | :---: |
|<img src='./assets/applications/human_detection.png' height="256px" >| <img src='./assets/applications/helmet_detection.png' height="256px">|
|[**口罩检测**](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_facemask/summary) |[**香烟检测**](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_cigarette/summary) | 
|<img src='./assets/applications/facemask_detection.png' height="256px">| <img src='./assets/applications/cigarette_detection.png' height="256px">|



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
