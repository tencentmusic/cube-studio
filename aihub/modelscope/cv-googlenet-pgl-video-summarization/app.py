import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_GOOGLENET_PGL_VIDEO_SUMMARIZATION_Model(Model):
    # 模型基础信息定义
    name='cv-googlenet-pgl-video-summarization'   # 该名称与目录名必须一样，小写
    label='PGL_SUM视频摘要-Web视频领域'
    describe="视频摘要，输入一段长视频，算法对视频进行镜头切割得到视频片段，评估视频帧的重要性，输出重要视频帧的帧号，根据帧号可以合成一段短视频（摘要视频）。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "11815"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_googlenet_pgl-video-summarization/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.video, name='video', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "",
            "input": {
                "video": "/mnt/workspace/.cache/modelscope/damo/cv_googlenet_pgl-video-summarization/video/video_category_test_video.mp4"
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
        
        self.p = pipeline('video-summarization', 'damo/cv_googlenet_pgl-video-summarization')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,video,**kwargs):
        result = self.p(video)

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

model=CV_GOOGLENET_PGL_VIDEO_SUMMARIZATION_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(video='/mnt/workspace/.cache/modelscope/damo/cv_googlenet_pgl-video-summarization/video/video_category_test_video.mp4')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()