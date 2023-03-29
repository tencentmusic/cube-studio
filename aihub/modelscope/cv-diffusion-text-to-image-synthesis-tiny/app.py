import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import time,cv2,numpy
import pysnooper
import os

class CV_DIFFUSION_TEXT_TO_IMAGE_SYNTHESIS_TINY_Model(Model):
    # 模型基础信息定义
    name='cv-diffusion-text-to-image-synthesis-tiny'   # 该名称与目录名必须一样，小写
    label='文本到图像生成扩散模型-中英文-通用领域-tiny'
    describe="文本到图像生成扩散模型-中英文-通用领域-tiny"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "2302"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_diffusion_text-to-image-synthesis_tiny/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.text, name='text', label='输入prompt',describe='输入prompt',default='',validators=Validator(max=75))
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[]

    # 训练的入口函数，将用户输入参数传递
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass


    # 加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('text-to-image-synthesis', 'damo/cv_diffusion_text-to-image-synthesis_tiny')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,text,**kwargs):
        result = self.p({'text': '中国山水画'})
        savePath = 'result/result_' + str(int(1000*time.time())) + '.jpg'
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        if os.path.exists(savePath):
            os.remove(savePath)

        cv2.imwrite(savePath, result['output_imgs'][0])
        back=[
            {
                "image": savePath
            }
        ]
        return back

model=CV_DIFFUSION_TEXT_TO_IMAGE_SYNTHESIS_TINY_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(text='中国山水画')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()

# 小模型 4G
# cpu推理速度 时间太久