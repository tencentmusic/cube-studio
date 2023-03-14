
视频人像抠图（Video human matting）是计算机视觉的经典任务，输入一个视频（图像序列），得到对应视频中人像的alpha图，其中alpha与分割mask不同，mask将视频分为前景与背景，取值只有0和1，而alpha的取值范围是0到1之间，返回数值代表透明度。VHM模型处理1080P视频每帧计算量为10.6G，参数量只有6.3M。

<div align=center>
<img src="resource/test_com.gif">
</div>

<div align=center>
<img src="resource/test_double.gif">
</div>

### 抠图系列模型

| [<img src="resource/imgseg.png" width="280px">](https://modelscope.cn/models/damo/cv_unet_image-matting/summary) | [<img src="resource/common.png" width="280px">](https://modelscope.cn/models/damo/cv_unet_universal-matting/summary) | [<img src="resource/videomat.png" width="280px">](https://modelscope.cn/models/damo/cv_effnetv2_video-human-matting/summary) |[<img src="resource/sky.png" width="280px">](https://modelscope.cn/models/damo/cv_hrnetocr_skychange/summary)|
|:--:|:--:|:--:|:--:| 
| [图像人像抠图](https://modelscope.cn/models/damo/cv_unet_image-matting/summary) | [通用抠图(支持商品、动物、植物、汽车等抠图)](https://modelscope.cn/models/damo/cv_unet_universal-matting/summary) | [视频人像抠图](https://modelscope.cn/models/damo/cv_effnetv2_video-human-matting/summary) | [天空替换(一键实现魔法换天空)](https://modelscope.cn/models/damo/cv_hrnetocr_skychange/summary) |


### 模型结构介绍
该模型由四部分构成：backbone、中间处理层、decoder和高分辨率处理层；

其中backbone是基于efficientnetv2实现，为了实现更小的计算量和更好的效果，我们对原网络模块进行了一定修改后重新与unet结构进行人像分割任务训练，结果的backbone参数作为预训练载入；
中间处理层使用ASPP模块作为基本单元，多层空洞卷积扩大感受野；
decoder部分将会逐步融合backbone各层特征，同时将原始图像特征作为一部分输入来引导网络学习，此外，我们将gru作为基础模块嵌入网络以便于处理时序相关信息；
高分辨率处理层基于DGF（Deep Guided Filter），实现对低分辨率后超分至原有分辨率，该层仍具有可学习参数，效果远优于直接上采样效果。

### 如何使用

在ModelScope框架上，提供输入视频和输出目录，即可以通过简单的Pipeline调用来使用视频人像抠图。

#### 代码范例

需要ModelScope版本≥1.1.1

```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_matting = pipeline(Tasks.video_human_matting, 
                       model='damo/cv_effnetv2_video-human-matting')
result_status = video_matting({'video_input_path':'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_matting_test.mp4',
                           'output_path':'matting_out.mp4'})
result = result_status[OutputKeys.MASKS]

```
- video\_input\_path 为输入视频的路径，示例代码为线上视频路径，可更改为本地视频路径
- output\_path 为输出视频的本地路径

正常情况下，输出路径会返回人像抠图的mask视频结果，算法result返回的是包含每帧narray格式结果的列表，同时打印字符串'matting process done'


### 训练数据集介绍
该模型训练可以同时使用分割数据集和matting数据集，可使用的数据集如COCO、HumanMatting等



### 模型局限性以及可能的偏差
受限于训练数据集，有可能产生一些偏差，请用户自行评测后决定如何使用。
