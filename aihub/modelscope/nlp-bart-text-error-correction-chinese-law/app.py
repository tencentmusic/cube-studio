import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_BART_TEXT_ERROR_CORRECTION_CHINESE_LAW_Model(Model):
    # 模型基础信息定义
    name='nlp-bart-text-error-correction-chinese-law'   # 该名称与目录名必须一样，小写
    label='BART文本纠错-中文-法律领域-large'
    describe="法研杯2022文书校对赛道冠军纠错模型（单模型）。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.png'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "1447"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/nlp_bart_text-error-correction_chinese-law/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='arg0', label='',describe='',default='',validators=Validator(max=100))
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "2012年、2013年收入统计表复印件各一份，欲证明被告未足额知府社保费用。"
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": "研发费用（不包括政府补助部分）当年占销售收入比例超过3%的，超过部分财政给予20％的奖励，最高不超过50万。"
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": "原告主张的未及时出具终止动合同证明的损失及律师费无依据，不同意赔偿。"
            }
        },
        {
            "label": "示例4",
            "input": {
                "arg0": "为保证甘肃省国家订货合同时顺利执行，按时履约，合同签订后，应将其副本抄送有关铁路和所在地工上行政管理部门。"
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下


    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('text-error-correction', 'damo/nlp_bart_text-error-correction_chinese-law')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
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

model=NLP_BART_TEXT_ERROR_CORRECTION_CHINESE_LAW_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='2012年、2013年收入统计表复印件各一份，欲证明被告未足额知府社保费用。')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()