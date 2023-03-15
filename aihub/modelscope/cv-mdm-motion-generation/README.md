
# 运动生成模型介绍
![a](description/a.gif)
![b](description/b.gif)

根据文本描述，自动生成人体的运动对很多行业都有重要的应用，例如动画制作，元宇宙以及机器人等。上图展示了模型的输入和输出效果
## 模型描述

![model](description/arch.png)


整个任务根据文字描述生成对应人体的运动, 模型是由一个MDM模型构成的扩散模型，左边MDM模型输入噪声和生成条件C，右边展示了扩展生成过程。

## 使用方式和范围
输入对人体运动的描述（英文），算法生成对应的人体运动数据。
输入文本样例：

```
a person is standing with both hands in front of them, then raises both arms up at the shoulder
a person walks forward then around off to the side
a person walks forward, shuffles to the left, the walks to the right
he puts leg up and down
a person is running from side ti side
a person fastly swimming forward
a person scrawling backwards slowly
a person is doing push ups
standing on one leg and swinging it
someone walking then sitting down in a chair
```

#### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#创建pipeline
motion_generation_pipeline = pipeline(Tasks.motion_generation, 'damo/cv_mdm_motion-generation')

#调用pipeline
result = motion_generation_pipeline('the person walked forward and is picking up his toolbox', output_video='demo.mp4')

print(f'motion generation result: {result}.')

```
输出：
```json
{"keypoints": np.array, "output_video": "demo.mp4"}
```
- `keypoints`: 人体关键点运动序列，shape 为`n_frame,n_joint,3`
- `output_video`: 输出可视化结果视频路径，如果调用pipeline时没有给出`output_video`参数，则自动生成临时路径并返回该临时路径
- 
  
## 数据评估以及结果
在公开数据集HumanML3D上评估生成结果

![HumanML3D性能](description/result.png)

## 引用
```BibTeX
@article{tevet2022human,
  title={Human Motion Diffusion Model},
  author={Tevet, Guy and Raab, Sigal and Gordon, Brian and Shafir, Yonatan and Bermano, Amit H and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2209.14916},
  year={2022}
}
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}
@INPROCEEDINGS{petrovich21actor,
  title     = {Action-Conditioned 3{D} Human Motion Synthesis with Transformer {VAE}},
  author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```


