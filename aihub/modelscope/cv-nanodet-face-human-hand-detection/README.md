
# 目标检测-人脸人体人手-通用领域

这是一个人脸、人体、人手三合一检测模型

## 模型描述

利用[NanoDet](https://github.com/RangiLyu/nanodet)，进行人脸、人体、人手的检测

## 使用方式和范围


### 如何使用

在ModelScope框架上，提供图片，得到识别的结果

#### 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

face_human_hand_detection = pipeline(Tasks.face_human_hand_detection, model='damo/cv_nanodet_face-human-hand-detection')
result_status = face_human_hand_detection({'input_path': 'data/test/images/face_human_hand_detection.jpg'})
labels = result_status[OutputKeys.LABELS]
boxes = result_status[OutputKeys.BOXES]
scores = result_status[OutputKeys.SCORES]

```

输出结果示例如下：

labels = [2, 1, 0]

boxes = [[78, 282, 240, 504], [127, 87, 332, 370], [0, 0, 367, 639]]

scores = [0.8202137351036072, 0.8987470269203186, 0.9679114818572998]

labels为类别，0代表人体，1代表人脸，2代表人手，

boxes为和labels对应的检测框的坐标，中间4个数字代表检测框的坐标，分别代表左上角的x，左上角的y，右下角的x，右下角的y

scores为对应的置信度分数


### 引用
```BibTeX
@misc{=nanodet,
    title={NanoDet-Plus: Super fast and high accuracy lightweight anchor-free object detection model.},
    author={RangiLyu},
    howpublished = {\url{https://github.com/RangiLyu/nanodet}},
    year={2021}
}
```