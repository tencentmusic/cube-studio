import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CRNN_OCR_RECOGNITION_GENERAL_DAMO_Model(Model):
    # 模型基础信息定义
    name='cv-crnn-ocr-recognition-general-damo'   # 该名称与目录名必须一样，小写
    label='读光-文字识别-CRNN模型-中英-通用领域'
    describe=""
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "1430"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_crnn_ocr-recognition-general_damo/summary"

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
        
        self.p = pipeline('ocr-recognition', 'damo/cv_crnn_ocr-recognition-general_damo')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
        text = result.get("text")
        back=[
            {
                "text": str(text),
            }
        ]
        return back

model=CV_CRNN_OCR_RECOGNITION_GENERAL_DAMO_Model()

# 测试后将此部分注释
#model.load_model()
#result = model.inference(image='test.jpg')  # 测试
#print(result)

# 测试后打开此部分
if __name__=='__main__':
     model.run()
#模型大小为46M,运行内存占用为436M,响应速度在两秒以内,没有GPU
#运行环境为腾讯云服务器	标准型S6 - 2核 4G,操作系统TencentOS Server 3.1 (TK4)
#识别字迹清晰的广告牌比较准确,识别手写字有识别错误
