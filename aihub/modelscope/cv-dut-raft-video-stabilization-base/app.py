import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_DUT_RAFT_VIDEO_STABILIZATION_BASE_Model(Model):
    # 模型基础信息定义
    name='cv-dut-raft-video-stabilization-base'   # 该名称与目录名必须一样，小写
    label='DUT-RAFT视频稳像'
    describe=""
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.gif'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "1757"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_dut-raft_video-stabilization_base/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.video, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "https://vigen-video.oss-cn-shanghai.aliyuncs.com/ModelScope/test/videos/regular_0.mp4?OSSAccessKeyId=LTAI5tR4AFwfzcCNtb8WXCXR&Expires=1682265190&Signature=IznE8bXg2u7g3cxE3MmzJ0E6Z14%3D"
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": "https://vigen-video.oss-cn-shanghai.aliyuncs.com/ModelScope/test/videos/moving_104c.mp4?OSSAccessKeyId=LTAI5tR4AFwfzcCNtb8WXCXR&Expires=1682265668&Signature=R5mxpw8VenabGkpaeZsbBoIIW74%3D"
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
        
        self.p = pipeline('video-stabilization', 'damo/cv_dut-raft_video-stabilization_base')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result =self.p(arg0)
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

model=CV_DUT_RAFT_VIDEO_STABILIZATION_BASE_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(arg0='https://vigen-video.oss-cn-shanghai.aliyuncs.com/ModelScope/test/videos/regular_0.mp4?OSSAccessKeyId=LTAI5tR4AFwfzcCNtb8WXCXR&Expires=1682265190&Signature=IznE8bXg2u7g3cxE3MmzJ0E6Z14%3D')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
