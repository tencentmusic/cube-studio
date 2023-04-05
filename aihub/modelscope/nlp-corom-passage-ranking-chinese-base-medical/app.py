import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_COROM_PASSAGE_RANKING_CHINESE_BASE_MEDICAL_Model(Model):
    # 模型基础信息定义
    name='nlp-corom-passage-ranking-chinese-base-medical'   # 该名称与目录名必须一样，小写
    label='CoROM语义相关性-中文-医疗领域-base'
    describe="基于CoROM-Base预训练模型的医疗领域中文语义相关性模型，模型以一个source sentence以及一个句子列表作为输入，最终输出source sentence与列表中每个句子的相关性得分（0-1，分数越高代表两者越相关）。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "4929"
    frameworks = ""
    doc = "https://modelscope.cn/models/damo/nlp_corom_passage-ranking_chinese-base-medical/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='source_sentence', label='',describe='',default='',validators=None),
        Field(type=Field_type.text_select, name='sentences_to_compare', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例0",
            "input": {
                "arg0": [
                    "上消化道出血手术大约多久"
                ],
                "arg1": [
                    "上消化道出血手术大约要2-3个小时左右。手术后应观察血压、体温、脉搏、呼吸的变化。污染被服应随时更换，以避免不良刺激。出血停止后按序给予温凉流质、半流质及易消化的软饮食。",
                    "胃出血一般住院30-60天。胃出血一般需要住院的时间需要注意根据情况来看，要看是胃溃疡引起，还是有无肝硬化门静脉高压引起的出血的情况，待消化道出血完全停止后病情稳定就可以出院，因此住院时间并不固定。"
                ]
            }
        },
        {
            "label": "示例1",
            "input": {
                "arg0": [
                    "大葱鲜姜紫菜汤治疗冠心病,汤里放盐吗?"
                ],
                "arg1": [
                    "冠心病患者应避免辛辣刺激，大葱和生姜都有一定的刺激性。然后对盐的摄入应保持在每天5~10克之间。",
                    "冠心病就是所谓的心脏病的一种。饮食主要是少吃油腻的，低盐饮食。",
                    "盐开水有一定的消炎作用。"
                ]
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": [
                    "孩子嘴里擦紫药水产生副作用了怎么办呢"
                ],
                "arg1": [
                    "病例分析:不要紫药水，有问题不好观察，现在不主张用。停药就好了。意见建议:多用白水漱口。",
                    "给宝宝服药治疗期间要注意饮食调理，多喝温开水，忌食辛辣刺激油腻生冷的饮食，不要喝碳酸饮料",
                    "你好，你孩子的情况考虑是消化不好引起的，建议服用鞣酸蛋白，胶囊治疗。"
                ]
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": [
                    "全身肌肉无力是挂神经内科还是免疫内科？"
                ],
                "arg1": [
                    "先挂神经内科查找无力原因。",
                    "若是和外伤或占位性疾患的情况是属于神经外科，而其他情况是内科。",
                    "强直性脊柱炎是风湿免疫病的一种。往往是到风湿免疫科看病的。"
                ]
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
        
        self.p = pipeline('text-ranking', 'damo/nlp_corom_passage-ranking_chinese-base-medical')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,source_sentence,sentences_to_compare,**kwargs):
        result = self.p({"source_sentence": source_sentence, "sentences_to_compare": sentences_to_compare})

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

model=NLP_COROM_PASSAGE_RANKING_CHINESE_BASE_MEDICAL_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='['上消化道出血手术大约多久']',arg1='['上消化道出血手术大约要2-3个小时左右。手术后应观察血压、体温、脉搏、呼吸的变化。污染被服应随时更换，以避免不良刺激。出血停止后按序给予温凉流质、半流质及易消化的软饮食。', '胃出血一般住院30-60天。胃出血一般需要住院的时间需要注意根据情况来看，要看是胃溃疡引起，还是有无肝硬化门静脉高压引起的出血的情况，待消化道出血完全停止后病情稳定就可以出院，因此住院时间并不固定。']')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()