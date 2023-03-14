
# 视频流深度与相机位姿估计算法介绍

## 任务
输入一段RGB视频流，深度与相机位姿估计算法将分析场景三维结构、输出图像对应的稠密深度图以及图像之间的相对相机位姿

## 模型描述

本模型基于**DRO: Deep Recurrent Optimizer for Structure-from-Motion**算法，是该算法的官方模型。

技术细节请见：

**DRO: Deep Recurrent Optimizer for Structure-from-Motion** <br />
Xiaodong Gu*, Weihao Yuan*, Zuozhuo Dai, Chengzhou Tang, Siyu Zhu, Ping Tan <br />
**[[Paper](https://arxiv.org/abs/2103.13201)]** |
**[[中文解读](https://zhuanlan.zhihu.com/p/372320845)]**  <br />

<p float="left">
  <img src="description/figs/demo_kitti.gif" width="400" />
  <img src="description/figs/demo_scannet.gif" width="400" /> 
</p>


## 如何使用

### 代码示例

```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import show_video_depth_estimation_result


task = 'video-depth-estimation'
model_id = 'damo/cv_dro-resnet18_video-depth-estimation_indoor'

input_location = 'data/test/videos/video_depth_estimation.mp4'
estimator = pipeline(Tasks.video_depth_estimation, model=model_id)
result = estimator(input_location)
show_video_depth_estimation_result(result[OutputKeys.DEPTHS_COLOR], 'out.mp4')
```

### 适用范围

默认输入图片的摄像机参数应与训练数据集（ScanNet）保持一直, 即分辨率为1296x968，内参为
```
1170.187988,         0.0,   647.750000
        0.0, 1170.187988,   483.750000
        0.0,         0.0,          0.0
```
如输入图像不一致，请将输入图片矫正为上述参数，否则会影响结果准确性


## 模型精度

| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | a1 | a2 | a3| SILog| L1_inv| rot_ang| t_ang| t_cm| 
| :--- | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|scannet_sup | 0.053 | 0.017 | 0.165 | 0.080 | 0.967 | 0.994 | 0.998| 0.078 | 0.033| 0.472| 9.297| 1.160|

## Bibtex
```
@article{gu2021dro,
  title={DRO: Deep Recurrent Optimizer for Structure-from-Motion},
  author={Gu, Xiaodong and Yuan, Weihao and Dai, Zuozhuo and Tang, Chengzhou and Zhu, Siyu and Tan, Ping},
  journal={arXiv preprint arXiv:2103.13201},
  year={2021}
}
```

## Acknowledgements
该项目中一些代码来自于 [packnet-sfm](https://github.com/TRI-ML/packnet-sfm) 和 [RAFT](https://github.com/princeton-vl/RAFT)，非常感谢他们开源了相关工作。
