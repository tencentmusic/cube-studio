import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CONVNEXTTINY_OCR_RECOGNITION_GENERAL_DAMO_Model(Model):
    # 模型基础信息定义
    name='cv-convnexttiny-ocr-recognition-general-damo'   # 该名称与目录名必须一样，小写
    label='读光-文字识别-行识别模型-中英-通用领域'
    describe="给定一张图片，识别出图中所含文字并输出字符串。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "37469"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-general_damo/summary"

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
        
        self.p = pipeline('ocr-recognition', 'damo/cv_convnextTiny_ocr-recognition-general_damo')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
        text = result.get('text')
        back=[
            {
                "text": str(text),
            }
        ]
        return back

model=CV_CONVNEXTTINY_OCR_RECOGNITION_GENERAL_DAMO_Model()

# 测试后将此部分注释
#model.load_model()
#result = model.inference(image='test.jpg')  # 测试
#print(result)

# 测试后打开此部分
if __name__=='__main__':
     model.run()
#模型大小为74M,运行内存占用为511M,响应速度在两秒以内,没有GPU
#运行环境为腾讯云服务器	标准型S6 - 2核 4G,操作系统TencentOS Server 3.1 (TK4)
#模型只能指标图片中一段文字,太长有识别缺失的情况,英文识别有单词缺少字母的情况,对于字体有要求
