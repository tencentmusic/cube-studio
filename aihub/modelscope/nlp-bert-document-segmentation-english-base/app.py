import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_BERT_DOCUMENT_SEGMENTATION_ENGLISH_BASE_Model(Model):
    # 模型基础信息定义
    name='nlp-bert-document-segmentation-english-base'   # 该名称与目录名必须一样，小写
    label='BERT文本分割-英文-通用领域'
    describe="该模型基于wiki-en公开语料训练，对未分割的长文本进行段落分割。提升未分割文本的可读性以及下游NLP任务的性能。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "1070"
    frameworks = ""
    doc = "https://modelscope.cn/models/damo/nlp_bert_document-segmentation_english-base/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='input', label='文本',describe='文本',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "input": "The Saint Alexander Nevsky Church was established in 1936 by Archbishop Vitaly (Maximenko) () on a tract of land donated by Yulia Martinovna Plavskaya.The initial chapel, dedicated to the memory of the great prince St. Alexander Nevsky (1220–1263), was blessed in May, 1936.The church building was subsequently expanded three times.In 1987, ground was cleared for the construction of the new church and on September 12, 1989, on the Feast Day of St. Alexander Nevsky, the cornerstone was laid and the relics of St. Herman of Alaska placed in the foundation.The imposing edifice, completed in 1997, is the work of Nikolaus Karsanov, architect and Protopresbyter Valery Lukianov, engineer.Funds were raised through donations.The Great blessing of the cathedral took place on October 18, 1997 with seven bishops, headed by Metropolitan Vitaly Ustinov, and 36 priests and deacons officiating, some 800 faithful attended the festivity.The old church was rededicated to Our Lady of Tikhvin.Metropolitan Hilarion (Kapral) announced, that cathedral will officially become the episcopal See of the Ruling Bishop of the Eastern American Diocese and the administrative center of the Diocese on September 12, 2014.At present the parish serves the spiritual needs of 300 members.The parochial school instructs over 90 boys and girls in religion, Russian language and history.The school meets every Saturday.The choir is directed by Andrew Burbelo.The sisterhood attends to the needs of the church and a church council acts in the administration of the community.The cathedral is decorated by frescoes in the Byzantine style.The iconography project was fulfilled by Father Andrew Erastov and his students from 1995 until 2001."
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
        
        self.p = pipeline('document-segmentation', 'damo/nlp_bert_document-segmentation_english-base')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,input,**kwargs):
        result = self.p(input)

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

model=NLP_BERT_DOCUMENT_SEGMENTATION_ENGLISH_BASE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(input='The Saint Alexander Nevsky Church was established in 1936 by Archbishop Vitaly (Maximenko) () on a tract of land donated by Yulia Martinovna Plavskaya.The initial chapel, dedicated to the memory of the great prince St. Alexander Nevsky (1220–1263), was blessed in May, 1936.The church building was subsequently expanded three times.In 1987, ground was cleared for the construction of the new church and on September 12, 1989, on the Feast Day of St. Alexander Nevsky, the cornerstone was laid and the relics of St. Herman of Alaska placed in the foundation.The imposing edifice, completed in 1997, is the work of Nikolaus Karsanov, architect and Protopresbyter Valery Lukianov, engineer.Funds were raised through donations.The Great blessing of the cathedral took place on October 18, 1997 with seven bishops, headed by Metropolitan Vitaly Ustinov, and 36 priests and deacons officiating, some 800 faithful attended the festivity.The old church was rededicated to Our Lady of Tikhvin.Metropolitan Hilarion (Kapral) announced, that cathedral will officially become the episcopal See of the Ruling Bishop of the Eastern American Diocese and the administrative center of the Diocese on September 12, 2014.At present the parish serves the spiritual needs of 300 members.The parochial school instructs over 90 boys and girls in religion, Russian language and history.The school meets every Saturday.The choir is directed by Andrew Burbelo.The sisterhood attends to the needs of the church and a church council acts in the administration of the community.The cathedral is decorated by frescoes in the Byzantine style.The iconography project was fulfilled by Father Andrew Erastov and his students from 1995 until 2001.')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()