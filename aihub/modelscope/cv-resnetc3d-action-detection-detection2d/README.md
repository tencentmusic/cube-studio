
# 动作检测模型介绍


## 模型描述
输入视频文件，输出该段时间内视频所包含的动作。算法内部每两秒均匀采样4帧输入到动作检测模型中，然后按设定时间步长滑动对整个视频的动作进行检测并返回结果。CUDA和CPU运行环境均支持。

模型结构如下所示：

![模型结构](description/arch.png)


## 使用方式和范围
如果您还没有安装 ffmpeg 库，先运行如下命令安装：
```bash
conda install ffmpeg
```
使用方式：
- 直接推理，输入视频文件或者帧序列

使用范围:
- 支持检测的动作：举手、吃喝、吸烟、打电话、玩手机、趴桌睡觉、跌倒、洗手、拍照。
- 输入要求：输入视频大于2秒，小于10秒（大于10秒需要修改video_length_limit参数）

目标场景:
- 摄像机固定不动，距离拍摄目标2-10米。


## 数据评估以及结果
在众包采集的行为数据上进行测试

| 动作/指标     | 举手   | 吃喝   | 吸烟   | 打电话  | 玩手机  | 趴桌睡觉 | 跌倒   | 洗手   | 拍照   |
|-----------|------|------|------|------|------|------|------|------|------|
| Precision | 0.96 | 0.83 | 0.79 | 0.83 | 0.83 | 0.94 | 0.87 | 0.78 | 0.81 |
| Recall    | 0.96 | 0.80 | 0.83 | 0.92 | 0.76 | 0.94 | 0.84 | 0.81 | 0.80 |

不同场景的数据集测试性能会出现差异

## 主要配置参数解释
- video_length_limit: 视频越长，算法耗时越长。为保证程序返回时间，仅处理前n秒的视频, 默认10秒
- pre_nms_thresh: 每个行为类别的置信度阈值，可根据需要调整，取值范围0-1之间

## 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

action_detection_pipeline = pipeline(Tasks.action_detection, 'damo/cv_ResNetC3D_action-detection_detection2d')
result = action_detection_pipeline('https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/action_detection_test_video.mp4')

print(f'action detection result: {result}.')
```
输出：
```json
 {'timestamps': [1, 3, 5], 
  'labels': ['吸烟', '吸烟', '吸烟'], 
  'scores': [0.7527753114700317, 0.753358006477356, 0.6880350708961487], 
  'boxes': [[547, 2, 1225, 719], [529, 8, 1255, 719], [584, 0, 1269, 719]]}
```
- timestamps：时间戳，表示输入视频中动作发生的时刻，单位为秒
- labels: 动作类别标签
- scores: 动作置信度，取值0-1之间
- boxes: 动作发生的空间位置，[x1,y1,x2,y2]

## 最佳实践
该检测模型是面向落地场景设计的，可用于家庭、工厂等监控相机场景。
### 阈值调整
模型默认返回结果的阈值为0.45，可通过传入参数`pre_nms_thresh`来改变返回结果的阈值
```python
action_detection_pipeline = pipeline(Tasks.action_detection, 'damo/cv_ResNetC3D_action-detection_detection2d',pre_nms_thresh=[0.3, 0.3, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]) #修改举手、吃喝两种动作的的阈值为0.3

```
`pre_nms_thresh`由9个参数组成，分别表示9种动作的阈值。可以先尽量调低`pre_nms_thresh`的值，在返回后再根据场景设定新的阈值对结果进行过滤。
### 视频处理长度调整
为了保证程序返回结果的等待时间，默认仅处理前10秒的视频，如果需要修改这一设定，传入参数`video_length_limit`，参数单位为秒。
```python
action_detection_pipeline = pipeline(Tasks.action_detection, 'damo/cv_ResNetC3D_action-detection_detection2d',pre_nms_thresh=[0.3, 0.3, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45], video_length_limit=15) #处理前15秒视频

```
### 运行加速
修改默认参数`op_num_threads`可以加速ONNX模型运行速度，该参数值默认为1，修改为CPU核心的个数可最大化运行速度。建议设为0，此时程序将根据系统配置自动选择合适的线程个数。
```python
action_detection_pipeline = pipeline(Tasks.action_detection, 'damo/cv_ResNetC3D_action-detection_detection2d',pre_nms_thresh=[0.3, 0.3, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45], video_length_limit=10, op_num_threads=0)

```

## 模型局限性以及可能的偏差

- 目前仅支持9种常用的动作类别检测
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试 



