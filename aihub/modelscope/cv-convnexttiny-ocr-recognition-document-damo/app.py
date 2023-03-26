import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CONVNEXTTINY_OCR_RECOGNITION_DOCUMENT_DAMO_Model(Model):
    # 模型基础信息定义
    name='cv-convnexttiny-ocr-recognition-document-damo'   # 该名称与目录名必须一样，小写
    label='读光-文字识别-行识别模型-中英-文档印刷体文本领域'
    describe="给定一张文档印刷体图片，识别出图中所含文字并输出字符串。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "13225"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-document_damo/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "test.jpg"
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
        
        self.p = pipeline('ocr-recognition', 'damo/cv_convnextTiny_ocr-recognition-document_damo')
    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result =  self.p(image)
        text=result.get('text')
        back=[
            {
                "text": str(text),
            }
        ]
        return back

model=CV_CONVNEXTTINY_OCR_RECOGNITION_DOCUMENT_DAMO_Model()

# 测试后将此部分注释
#model.load_model()
#result = model.inference(image='test.jpg')  # 测试
#print(result)

# 测试后打开此部分
if __name__=='__main__':
     model.run()

#模型大小74M,内存占用544M,识别图片响应在2秒内
#模型识别的图片中只能有一行文字,多行不能识别,识别英文有单词缺少字母的情况,
#对于有些字体的汉字识别会有识别错误或识别缺失的情况
