import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CSPNET_VIDEO_OBJECT_DETECTION_STREAMYOLO_Model(Model):
    # 模型基础信息定义
    name='cv-cspnet-video-object-detection-streamyolo'   # 该名称与目录名必须一样，小写
    label='StreamYOLO实时视频目标检测-自动驾驶领域'
    describe="实时视频目标检测模型"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "4907"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_cspnet_video-object-detection_streamyolo/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.video, name='video', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "video": "/mnt/workspace/.cache/modelscope/damo/cv_cspnet_video-object-detection_streamyolo/res/test_vod_00.mp4"
            }
        },
        {
            "label": "示例2",
            "input": {
                "video": "/mnt/workspace/.cache/modelscope/damo/cv_cspnet_video-object-detection_streamyolo/res/test_vod_01.mp4"
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
        
        self.p = pipeline('video-object-detection', 'damo/cv_cspnet_video-object-detection_streamyolo')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,video,**kwargs):
        result = self.p(video)
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

model=CV_CSPNET_VIDEO_OBJECT_DETECTION_STREAMYOLO_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(video='/mnt/workspace/.cache/modelscope/damo/cv_cspnet_video-object-detection_streamyolo/res/test_vod_00.mp4')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
