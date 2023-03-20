import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os,cv2

import random

class CV_CARTOON_STABLE_DIFFUSION_CLIPART_Model(Model):
    # 模型基础信息定义
    name='cv-cartoon-stable-diffusion-clipart'   # 该名称与目录名必须一样，小写
    label='卡通系列文生图模型-剪贴画'
    describe=""
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "161"
    frameworks = "PyTorch"
    doc = "https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_clipart/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.text, name='text', label='输入prompt',describe='输入prompt',default='',validators=Validator(max=75))
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "text": "archer style, a portrait painting of Johnny Depp"
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

        self.p = pipeline('text-to-image-synthesis', 'damo/cv_cartoon_stable_diffusion_clipart')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,text,**kwargs):
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        self.p.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.p.pipeline.scheduler.config)

        result = self.p({'text':'archer style, a portrait painting of '+text})
        print(result)

        save_path='result/result'+str(random.randint(5,5000))+'.jpg'
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        cv2.imwrite(save_path, result['output_imgs'][0])
        back=[
            {
                "image": save_path
            }
        ]
        return back

model=CV_CARTOON_STABLE_DIFFUSION_CLIPART_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(text='archer style, a portrait painting of Johnny Depp')  # 测试



# 测试后打开此部分
if __name__=='__main__':
    model.run()
