import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_ALEX_VIDEO_SINGLE_OBJECT_TRACKING_SIAMFC_Model(Model):
    # 模型基础信息定义
    name='cv-alex-video-single-object-tracking-siamfc'   # 该名称与目录名必须一样，小写
    label='Siamfc视频单目标跟踪-通用领域-S'
    describe=""
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "2830"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_alex_video-single-object-tracking_siamfc/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.video, name='dog', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "dog": "http://dmshared.oss-cn-hangzhou.aliyuncs.com/ljp/maas/sot_demo_resouce/dog.mp4?OSSAccessKeyId=LTAI5tC7NViXtQKpxFUpxd3a&Expires=1706482434&Signature=7e3npX5OdADQV%2FV7JF970WsPnDI%3D"
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
        
        self.p = pipeline('video-single-object-tracking', 'damo/cv_alex_video-single-object-tracking_siamfc')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,dog,**kwargs):
        result = self.p(dog)
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

model=CV_ALEX_VIDEO_SINGLE_OBJECT_TRACKING_SIAMFC_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(dog='http://dmshared.oss-cn-hangzhou.aliyuncs.com/ljp/maas/sot_demo_resouce/dog.mp4?OSSAccessKeyId=LTAI5tC7NViXtQKpxFUpxd3a&Expires=1706482434&Signature=7e3npX5OdADQV%2FV7JF970WsPnDI%3D')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
