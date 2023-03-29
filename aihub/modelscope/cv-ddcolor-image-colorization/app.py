import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import time
import pysnooper
import os

class CV_DDCOLOR_IMAGE_COLORIZATION_Model(Model):
    # 模型基础信息定义
    name='cv-ddcolor-image-colorization'   # 该名称与目录名必须一样，小写
    label='DDColor图像上色'
    describe="DDColor 是最新的图像上色算法，输入一张黑白图像，返回上色处理后的彩色图像，并能够实现自然生动的上色效果。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "14680"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_ddcolor_image-colorization/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "/mnt/workspace/.cache/modelscope/damo/cv_ddcolor_image-colorization/resources/demo.jpg"
            }
        },
        {
            "label": "示例2",
            "input": {
                "image": "/mnt/workspace/.cache/modelscope/damo/cv_ddcolor_image-colorization/resources/demo2.jpg"
            }
        },
        {
            "label": "示例3",
            "input": {
                "image": "/mnt/workspace/.cache/modelscope/damo/cv_ddcolor_image-colorization/resources/demo3.jpg"
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
        
        self.p = pipeline('image-colorization', 'damo/cv_ddcolor_image-colorization')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
        savePath = 'result/result_' + str(int(1000*time.time())) + '.jpg'
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        if os.path.exists(savePath):
            os.remove(savePath)
        cv2.imwrite(savePath, result[OutputKeys.OUTPUT_IMG])
        back=[
            {
                "image": savePath
            }
        ]
        return back

model=CV_DDCOLOR_IMAGE_COLORIZATION_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(image='/mnt/workspace/.cache/modelscope/damo/cv_ddcolor_image-colorization/resources/demo.jpg')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()


# 模型大小 9000M
# cpu运行时长 3s钟