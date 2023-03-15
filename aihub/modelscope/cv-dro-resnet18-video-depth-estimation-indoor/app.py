import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_DRO_RESNET18_VIDEO_DEPTH_ESTIMATION_INDOOR_Model(Model):
    # 模型基础信息定义
    name='cv-dro-resnet18-video-depth-estimation-indoor'   # 该名称与目录名必须一样，小写
    label='循环神经优化器-视频流深度和相机轨迹估计'
    describe="我们提出zero-order的循环神经网络优化器（DRO）, 不需要求解梯度, 直接利用神经网络来预测下次更新的方向和步长。将优化目标cost，放入到神经网络中，每次迭代都会参考之前尝试的历史信息，从而给出更加精准的预测。也就是说，如果错误的预测值，就会使得cost变大，正确的预测值会使得cost变小，在不断尝试中，神经网络学习到了如何使得cost变小。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "216"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_dro-resnet18_video-depth-estimation_indoor/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.video, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_depth_estimation.mp4"
            }
        }
    ]

    # 训练的入口函数，将用户输入参数传递
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass


    # 加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('video-depth-estimation', 'damo/cv_dro-resnet18_video-depth-estimation_indoor')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)
        back=[
            {
                "image": 'result/aa.jpg',
                "text": '结果文本',
                "video": 'result/aa.mp4',
                "audio": 'result/aa.mp3',
                "markdown":''
            }
        ]
        return back

model=CV_DRO_RESNET18_VIDEO_DEPTH_ESTIMATION_INDOOR_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(arg0='https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_depth_estimation.mp4')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
