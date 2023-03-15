
# 动作识别模型介绍


## 模型描述
Patch Shift Transformers(PST) 是在2D Swin-Transformer的基础上，增加temporal建模能力，使网络具备视频时空特征学习能力。而这一操作几乎不增加额外参数。具体地，通过shift不同帧之间的patch, 然后在每帧内部分别进行self-attention 运算，这样使用2D的self-attention计算量来进行视频的时空特征建模，论文原文[链接](https://readpaper.com/paper/4650578659522920449)。

PatchShift示意图：


![模型结构](description/patchshift.png)


## 使用方式和范围

使用方式：
- 在公开数据集Kinetics400支持的标签上进行短视频分类，如果需要进行日常动作检测(如跌倒检测、吸烟检测等)，可使用[日常动作检测模型](https://modelscope.cn/models/damo/cv_ResNetC3D_action-detection_detection2d/summary)。


使用范围:
- 适合视频领域的动作识别检测，分辨率在224x224以上，输入视频长度10s以内

目标场景:
- 视频中的动作识别，比如体育、影视、直播等


#### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#创建pipeline
action_recognition_pipeline = pipeline(Tasks.action_recognition, 'damo/cv_pathshift_action-recognition')

#运行pipeline,输入视频的本地路径或者网络地址均可
result = action_recognition_pipeline('http://viapi-test.oss-cn-shanghai.aliyuncs.com/viapi-3.0domepic/facebody/RecognizeAction/RecognizeAction-video2.mp4')

print(f'action recognition result: {result}.')

```
输出：
```json
{'labels': 'abseiling'}
```
- labels: 英文类别名称

### 数据评估以及结果
在Something-Something V1 & V2，Kinetics400数据集上的模型性能：
| Dataset | Model  | Top@1 | Top@5 | 
| ------------ | ------------ | ------------ | ------------ |
| Sthv1 | PST-Tiny | 54.0 | 82.3 | 
| Sthv1 | PST-Base  | 58.3 | 83.9 | 
| Sthv2 | PST-Tiny | 67.9 | 90.8 | 
| Sthv2 | PST-Base  | 69.8 | 93.0 | 
| K400 | PST-Tiny  | 78.6 | 93.5 |
| K400 | PST-Base  | 82.5 | 95.6 | 

更多模型训练和测试细节可参考论文和开源[代码](https://github.com/MartinXM/TPS)。
### 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的论文：

```BibTeX
@article{xiang2022tps,
  title={Spatiotemporal Self-attention Modeling with Temporal Patch Shift for Action Recognition},
  author={Wangmeng Xiang, Chao Li, Biao Wang, Xihan Wei, Xian-Sheng Hua, Lei Zhang},
  journal={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

 