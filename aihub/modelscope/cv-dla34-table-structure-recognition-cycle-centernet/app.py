import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_DLA34_TABLE_STRUCTURE_RECOGNITION_CYCLE_CENTERNET_Model(Model):
    # 模型基础信息定义
    name='cv-dla34-table-structure-recognition-cycle-centernet'   # 该名称与目录名必须一样，小写
    label='读光-表格结构识别-有线表格'
    describe="有线表格结构识别，输入图像，检测出单元格bbox并将其拼接起来得到精准而完整的表格。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "3190"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_dla34_table-structure-recognition_cycle-centernet/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/table_recognition.jpg"
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
        
        self.p = pipeline('table-recognition', 'damo/cv_dla34_table-structure-recognition_cycle-centernet')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
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

model=CV_DLA34_TABLE_STRUCTURE_RECOGNITION_CYCLE_CENTERNET_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(image='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/table_recognition.jpg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
