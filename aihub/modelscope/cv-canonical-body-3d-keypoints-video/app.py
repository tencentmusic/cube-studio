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
        # print(result)

        back=[
            {
                "keypoints": str(result['keypoints']),
                "timestamp": str(result['timestamps']),
                "video": result['output_video']
            }
        ]


        return back

model=CV_CANONICAL_BODY_3D_KEYPOINTS_VIDEO_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(arg0='https://dmshared.oss-cn-hangzhou.aliyuncs.com/maas/test/video/Walking.54138969.h264.mp4')  # 测试
# print(result)

# 测试后打开此部分，已完成
if __name__=='__main__':
    # 1. 模型输出的视频是火柴棍形式，不是原视频加描线；
    # 2. 模型不能采帧，因为需要连续帧来判断3D的动作；
    # 3. 如果想输出更长的视频对应的人物动作，需要配置/mnt文件下的configuration.json的model.INPUT.MAX_FRAME，1秒30帧，如果输出5s的结果，就设置150帧
    # 4. 视频的输出结果没有位置信息，是原地的动作识别，需要配合一个人体框识别才能输出完整的位置+动作信息；
    # 5. 模型非常慢，识别5s的动作，需要57分钟，最好是能上GPU，GPU是否可行，目前不知道。
    model.run()
