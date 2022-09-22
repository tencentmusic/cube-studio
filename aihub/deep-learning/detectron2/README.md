
前几天大家是不是都刷到了下面这个视频

博主本来想证明自己背景是真的，结果引来网友恶搞，纷纷换成各种各样的背景来“打假”。

今天咱也凑个热闹，用 AI 技术自动替换背景。

看下替换后的效果


思路并不难，我们先从原视频将人物分离出来，再将分离出来的人物“贴”到新背景视频中即可。

从视频中分类人物用到关键技术的是计算机视觉中的实例分割，比如，人脸检测

目标检测是通过矩形框标注检测的目标，相对容易。而实例分割是一项更精细的工作，因为需要对每个像素点分类，物体的轮廓是精准勾勒的。

我们今天用 detectron2 做实例分割，它是Facebook AI研究院开源的项目，功能强大，使用简单。

# 1. 安装

detectron2 支持在Linux和macOS系统上安装，Linux可以直接通过pip安装，而mac只能通过源码编译的方式安装，建议大家用Linux。

支持GPU和CPU运行，我使用的是3090显卡、CUDA11.1、Pytorch1.8。

# 2. 运行

我们用官方提供的预训练模型对图片做实例分割。


## 2.1 加载模型配置文件
```bash
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
model_cfg_file = model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.merge_from_file(model_cfg_file)
```

COCO-InstanceSegmentation代表用coco数据集训练的实例分割模型。

mask_rcnn_R_50_FPN_3x.yaml是模型训练用到的配置信息。

从下图也可以看到，detectron2除了提供实例分割模型，还提供目标检测、关键点检测等模型，还是比较全面的。

## 2.2 加载模型
```bash
model_weight_url = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.WEIGHTS = model_weight_url
```

mask_rcnn_R_50_FPN_3x.yaml文件中存放了预训练模型的url。当进行实例分割时，程序会自动从url处将模型下载到本地。存放的位置为：

但程序自动下载的方式可能会比较慢，这时候你可以用迅雷自己下载模型文件，放到对应的路径中即可。

2.3 实例分割

首先，读取一张图片，图片大小480 * 640。
```bash
img = cv2.imread('./000000439715.jpg')
```

实例化DefaultPredictor对象
```bash
from detectron2.engine import DefaultPredictor
predictor = DefaultPredictor(cfg)
```

对图片进行实例分割
```bash
out = predictor(img)
```

out变量中存放的是分割出来每个目标的类别id、检测框和目标遮罩。

out["instances"].pred_classes获取目标的类别id

这里一共检测到了 15 个目标，在配置文件中可以找到类别id和类别名称的映射关系。其中，0代表人，17代表马。

out["instances"].pred_masks获取目标的遮罩，我们取单个目标的遮罩研究一下它的用处。


可以看到，它的取值是布尔类型，并且shape和图片大小一样。

所以，遮罩是实例分割的结果，里面每个元素对应图片一个像素，取值为True代表该像素是检测出来的目标像素。

因此，我们可以通过遮罩给目标加上一层不透明度，从而把目标精确标注出来。
```bash
img_copy = img.copy()

alpha = 0.8
color = np.array([0, 255, 0])

img_copy[mask > 0, :] = img_copy[mask > 0, :] * alpha + color * (1-alpha)
```

上述给目标加上一层绿色的不透明度，效果如下：

可以看到，骑在马上的人已经被标注出来了。

# 3. 自动合成背景

有了上面的基础，我们就很容易合成视频了。

读取原视频每一帧中将人物分割出来，将分割出来的人物直接覆盖到新背景视频中对应的帧即可。

核心代码

```bash
# 读取原视频
ret, frame_src = cap.read()

# 读取新背景视频
ret, frame_bg = cap_bg.read()
# 调整背景尺寸跟原视频一样
frame_bg = cv2.resize(frame_bg, sz)

# 分割原视频人物
person_mask = instance_seg.predict(frame_src)
# 合成
frame_bg[person_mask, :] = frame_src[person_mask, :]
```

另外，这次我们想要检测的人正好预训练模型提供了。如果你想检测新的目标，就需要自己标注、训练模型。


