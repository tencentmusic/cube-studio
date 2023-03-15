import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CANONICAL_BODY_3D_KEYPOINTS_VIDEO_Model(Model):
    # 模型基础信息定义
    name='cv-canonical-body-3d-keypoints-video'   # 该名称与目录名必须一样，小写
    label='人体关键点检测-通用领域-3D'
    describe="输入一段单人视频，实现端到端的3D人体关键点检测，输出视频中每一帧的3D人体关键点坐标。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "4617"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_canonical_body-3d-keypoints_video/summary"

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
                "arg0": "https://dmshared.oss-cn-hangzhou.aliyuncs.com/maas/test/video/Walking.54138969.h264.mp4"
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
        
        self.p = pipeline('body-3d-keypoints', 'damo/cv_canonical_body-3d-keypoints_video')

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

model=CV_CANONICAL_BODY_3D_KEYPOINTS_VIDEO_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(arg0='https://dmshared.oss-cn-hangzhou.aliyuncs.com/maas/test/video/Walking.54138969.h264.mp4')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
