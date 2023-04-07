import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_REALBASICVSR_VIDEO_SUPER_RESOLUTION_VIDEOLQ_Model(Model):
    # 模型基础信息定义
    name='cv-realbasicvsr-video-super-resolution-videolq'   # 该名称与目录名必须一样，小写
    label='RealBasicVSR视频超分辨率'
    describe="长期传播会在轻度退化的情况下提高性能，但严重的退化可能会通过传播被放大，从而损害输出视频的质量。为了平衡细节合成和退化抑制之间的权衡，RealBasicVSR引入了图像预清理模块，在质量和效率上都优于现有方法。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.png'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "3301"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_realbasicvsr_video-super-resolution_videolq/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.video, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/000.mp4"
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/047.mp4"
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
        
        self.p = pipeline('video-super-resolution', 'damo/cv_realbasicvsr_video-super-resolution_videolq')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
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

model=CV_REALBASICVSR_VIDEO_SUPER_RESOLUTION_VIDEOLQ_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/000.mp4')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()