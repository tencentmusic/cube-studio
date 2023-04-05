import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_BERT_DOCUMENT_SEGMENTATION_CHINESE_BASE_Model(Model):
    # 模型基础信息定义
    name='nlp-bert-document-segmentation-chinese-base'   # 该名称与目录名必须一样，小写
    label='BERT文本分割-中文-通用领域'
    describe="该模型基于wiki-zh公开语料训练，对未分割的长文本进行段落分割。提升未分割文本的可读性以及下游NLP任务的性能。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "7162"
    frameworks = ""
    doc = "https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/summary"

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
                "input": "近年来，随着端到端语音识别的流行，基于Transformer结构的语音识别系统逐渐成为了主流。然而，由于Transformer是一种自回归模型，需要逐个生成目标文字，计算复杂度随着目标文字数量线性增加，限制了其在工业生产中的应用。针对Transoformer模型自回归生成文字的低计算效率缺陷，学术界提出了非自回归模型来并行的输出目标文字。根据生成目标文字时，迭代轮数，非自回归模型分为：多轮迭代式与单轮迭代非自回归模型。其中实用的是基于单轮迭代的>非自回归模型。对于单轮非自回归模型，现有工作往往聚焦于如何更加准确的预测目标文字个数，如CTC-enhanced采用CTC预测输出文字个数，尽管如此，考虑到现实应用中，语速、口音、静音以及噪声等因素的影响，如何准确的预测目标文字个数以及抽取目标文字对应的声学隐变量仍然是一个比较大的挑战；另外一方面，我们通过对比自回归模型与单轮非自回归模型在工业大数据上的错误类型（如下图所示，AR与vanilla NAR），发现，相比于自回归模型，非自回归模型，在预测目标文字个数方面差距较小，但是替换错误显著的增加，我们认为这是由于单轮非自回归模型中条件独立假设导致的语义信息丢失。于此同时，目前非自回归模型主要停留在学术验证阶段，还没有工业大数据上的相关实验与结论。"
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
        
        self.p = pipeline('document-segmentation', 'damo/nlp_bert_document-segmentation_chinese-base')

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

model=NLP_BERT_DOCUMENT_SEGMENTATION_CHINESE_BASE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(input='近年来，随着端到端语音识别的流行，基于Transformer结构的语音识别系统逐渐成为了主流。然而，由于Transformer是一种自回归模型，需要逐个生成目标文字，计算复杂度随着目标文字数量线性增加，限制了其在工业生产中的应用。针对Transoformer模型自回归生成文字的低计算效率缺陷，学术界提出了非自回归模型来并行的输出目标文字。根据生成目标文字时，迭代轮数，非自回归模型分为：多轮迭代式与单轮迭代非自回归模型。其中实用的是基于单轮迭代的>非自回归模型。对于单轮非自回归模型，现有工作往往聚焦于如何更加准确的预测目标文字个数，如CTC-enhanced采用CTC预测输出文字个数，尽管如此，考虑到现实应用中，语速、口音、静音以及噪声等因素的影响，如何准确的预测目标文字个数以及抽取目标文字对应的声学隐变量仍然是一个比较大的挑战；另外一方面，我们通过对比自回归模型与单轮非自回归模型在工业大数据上的错误类型（如下图所示，AR与vanilla NAR），发现，相比于自回归模型，非自回归模型，在预测目标文字个数方面差距较小，但是替换错误显著的增加，我们认为这是由于单轮非自回归模型中条件独立假设导致的语义信息丢失。于此同时，目前非自回归模型主要停留在学术验证阶段，还没有工业大数据上的相关实验与结论。')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()