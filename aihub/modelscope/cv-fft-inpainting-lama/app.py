import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_FFT_INPAINTING_LAMA_Model(Model):
    # 模型基础信息定义
    name='cv-fft-inpainting-lama'   # 该名称与目录名必须一样，小写
    label='LaMa图像填充'
    describe="针对自然图片进行填充恢复，支持高分辨率图像的输入，同时支持在线refinement，使得高分辨率图片恢复出更加真实的内容细节"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "1003694"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_fft_inpainting_lama/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='img', label='',describe='',default='',validators=None),
        Field(type=Field_type., name='mask', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting.png",
                "mask":"https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting_mask.png"
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
        
        self.p = pipeline('image-inpainting', 'damo/cv_fft_inpainting_lama')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,img,mask,**kwargs):
        result = self.p({'img':img,'mask':mask})
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

model=CV_FFT_INPAINTING_LAMA_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(image='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting.png',mask='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting_mask.png')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
