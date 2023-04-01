from typing import List
from pydantic import BaseModel
import numpy
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import pysnooper

class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Action(BaseModel):
    label: str
    score: float
    box: Box
    timestamp: int

class ActionDetectionResult(BaseModel):
    actions: List[Action]
class CV_RESNETC3D_ACTION_DETECTION_DETECTION2D_Model(Model):
    # 模型基础信息定义
    name='cv-resnetc3d-action-detection-detection2d'   # 该名称与目录名必须一样，小写
    label='日常动作检测'
    describe="输入视频文件，输出该段时间内视频所包含的动作，当前支持9中常见动作识别"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.png'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "6875"
    frameworks = "ONNX"
    doc = "https://modelscope.cn/models/damo/cv_ResNetC3D_action-detection_detection2d/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.video, name='video', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例0",
            "input": {
                "video": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/action_detection_test_video.mp4"
            }
        },
        {
            "label": "示例1",
            "input": {
                "video": "/mnt/workspace/.cache/modelscope/damo/cv_ResNetC3D_action-detection_detection2d/description/call.mp4"
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下


    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.p = pipeline('action-detection', 'damo/cv_ResNetC3D_action-detection_detection2d',
                                             pre_nms_thresh=[0.3, 0.3, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                                             video_length_limit=10, op_num_threads=0)

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs) -> numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self, video, **kwargs):
        result = self.p(video)
        print(str(result))
        actions = []
        for idx in range(len(result['scores'])):
            timestamp = result['timestamps'][idx]
            label = result['labels'][idx]
            score = result['scores'][idx]
            (x1, y1, x2, y2) = result['boxes'][idx]
            box = Box(x1=x1, y1=y1, x2=x2, y2=y2)
            actions.append(Action(label=label, score=score, box=box, timestamp=timestamp))
        back = [
            {
                "text": str(actions),
            }
        ]
        return back

model=CV_RESNETC3D_ACTION_DETECTION_DETECTION2D_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(video='https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/action_detection_test_video.mp4')  # 测试
# print(result)

# # 模型启动web时使用
if __name__=='__main__':
    model.run()

# 模型大小：99MB
# 模型效果：近距离识别率较高
# 推理性能: 5s内
# 模型占用内存/推理服务占用内存/gpu占用显存：7MB/1GB/0GB
# 巧妙使用方法：chrom/safari PC验证通过、IOS失败