import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_EFFNETV2_VIDEO_HUMAN_MATTING_Model(Model):
    # 模型基础信息定义
    name='cv-effnetv2-video-human-matting'   # 该名称与目录名必须一样，小写
    label='视频人像抠图模型-通用领域'
    describe="输入一段视频，返回视频中人像的alpha序列"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "1488"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_effnetv2_video-human-matting/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.video, name='video_input_path', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "",
            "input": {
                "video_input_path": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_matting_test.mp4"
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
        
        self.p = pipeline('video-human-matting', 'damo/cv_effnetv2_video-human-matting')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,video_input_path,**kwargs):
        result = self.p({'video_input_path':video_input_path})
        pass
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

model=CV_EFFNETV2_VIDEO_HUMAN_MATTING_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(video_input_path='https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_matting_test.mp4')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
