import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os,random,cv2

class CV_CARTOON_STABLE_DIFFUSION_ILLUSTRATION_Model(Model):
    # 模型基础信息定义
    name='cv-cartoon-stable-diffusion-illustration'   # 该名称与目录名必须一样，小写
    label='卡通系列文生图模型-漫画风'
    describe=""
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "90"
    frameworks = "PyTorch"
    doc = "https://modelscope.cn/models/damo/cv_cartoon_stable_diffusion_illustration/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.text_select, name='choice_t', label='选择生成人还是物体景象', describe='', default='人',choices=['人','物体景象'],validators=Validator(max=1)),
        Field(type=Field_type.text, name='text', label='输入关键词',describe='目前仅支持英文文本提示词输入，可输入人物、物体、场景,如Johnny Depp、cat、supermarket',default='Johnny Depp',validators=Validator(max=75))
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "text": "Johnny Depp"
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
        
        self.p = pipeline('text-to-image-synthesis', 'damo/cv_cartoon_stable_diffusion_illustration')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,choice_t,text,**kwargs):
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        self.p.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.p.pipeline.scheduler.config)

        #archer type,sks type
        describe_t = 'sks style, a portrait painting of ' if choice_t=='人' else 'sks style, a painting of '

        result = self.p({'text':describe_t+text})
        print(result)

        save_path='result/result'+str(random.randint(5,5000))+'.jpg'
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        print(save_path)
        cv2.imwrite(save_path, result['output_imgs'][0])
        back=[
            {
                "image": save_path
            }
        ]
        return back


model=CV_CARTOON_STABLE_DIFFUSION_ILLUSTRATION_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(text='sks style, a portrait painting of Johnny Depp')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()
