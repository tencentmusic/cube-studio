import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_DDSAR_FACE_DETECTION_ICLR23_DAMOFD_34G_Model(Model):
    # 模型基础信息定义
    name='cv-ddsar-face-detection-iclr23-damofd-34g'   # 该名称与目录名必须一样，小写
    label='DamoFD人脸检测关键点模型-34G'
    describe="给定一张图片，返回图片中人脸区域的位置和五点关键点。针对如何设计可以预测stage-level表征能力的精度预测器，DamoFD从刻画network expressivity的角度出发，提出了SAR-score来无偏的刻画stage-wise network expressivity，进而Auto搜索了适合人脸检测的backbone结构。后续被ICLR23接收。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "16"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_ddsar_face-detection_iclr23-damofd-34G/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/mog_face_detection.jpg"
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
        
        self.p = pipeline('face-detection', 'damo/cv_ddsar_face-detection_iclr23-damofd-34G')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result= self.p(image)
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

model=CV_DDSAR_FACE_DETECTION_ICLR23_DAMOFD_34G_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(image='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/mog_face_detection.jpg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
