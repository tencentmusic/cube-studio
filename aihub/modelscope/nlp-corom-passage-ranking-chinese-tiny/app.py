import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_COROM_PASSAGE_RANKING_CHINESE_TINY_Model(Model):
    # 模型基础信息定义
    name='nlp-corom-passage-ranking-chinese-tiny'   # 该名称与目录名必须一样，小写
    label='CoROM语义相关性-中文-通用领域-tiny'
    describe="基于ROM-tiny预训练模型的通用领域中文语义相关性模型，模型以一个source sentence以及一个句子列表作为输入，最终输出source sentence与列表中每个句子的相关性得分（0-1，分数越高代表两者越相关）。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "570"
    frameworks = ""
    doc = "https://modelscope.cn/models/damo/nlp_corom_passage-ranking_chinese-tiny/summary"

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
                    "功和功率的区别"
                ],
                "arg1": [
                    "功反映做功多少，功率反映做功快慢。",
                    "什么是有功功率和无功功率?无功功率有什么用什么是有功功率和无功功率?无功功率有什么用电力系统中的电源是由发电机产生的三相正弦交流电,在交>流电路中,由电源供给负载的电功率有两种;一种是有功功率,一种是无功功率.",
                    "优质解答在物理学中,用电功率表示消耗电能的快慢．电功率用P表示,它的单位是瓦特（Watt）,简称瓦（Wa）符号是W.电流在单位时间内做的功叫做电功率 以灯泡为例,电功率越大,灯泡越亮.灯泡的亮暗由电功率（实际功率）决定,不由通过的电流、电压、电能决定!"
                ]
            }
        },
        {
            "label": "示例1",
            "input": {
                "arg0": [
                    "什么是桥"
                ],
                "arg1": [
                    "由全国科学技术名词审定委员会审定的科技名词“桥”的定义为：跨越河流、山谷、障碍物或其他交通线而修建的架空通道",
                    "桥是一种连接两岸的建筑",
                    "转向桥，是指承担转向任务的车桥。一般汽车的前桥是转向桥。四轮转向汽车的前后桥，都是转向桥。",
                    "桥梁艺术研究桥梁美学效果的艺术。起源于人类修建桥梁的实践。作为一种建筑，由于功能不同、使用材料有差别，桥梁表现为不同的结构和形式。基本的有拱桥、梁桥和吊桥。"
                ]
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": [
                    "福鼎在哪个市"
                ],
                "arg1": [
                    "福鼎是福建省宁德市福鼎市。",
                    "福鼎位于福建省东北部，东南濒东海，水系发达，岛屿星罗棋布。除此之外，福鼎地貌丰富，著名的太姥山就在福鼎辖内，以峰险、石奇、洞幽、雾幻四绝著称于世。",
                    "福鼎市人民政府真诚的欢迎国内外朋友前往迷人的太姥山观光、旅游。",
                    "福建省福鼎市桐山街道,地处福鼎城区,地理环境优越,三面环山,东临大海,境内水陆交通方便,是福鼎政治、经济、文化、交通中心区。"
                ]
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": [
                    "吃完海鲜可以喝牛奶吗?"
                ],
                "arg1": [
                    "不可以，早晨喝牛奶不科学",
                    "吃了海鲜后是不能再喝牛奶的，因为牛奶中含得有维生素C，如果海鲜喝牛奶一起服用会对人体造成一定的伤害",
                    "吃海鲜是不能同时喝牛奶吃水果，这个至少间隔6小时以上才可以。",
                    "吃海鲜是不可以吃柠檬的因为其中的维生素C会和海鲜中的矿物质形成砷"
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
        
        self.p = pipeline('text-ranking', 'damo/nlp_corom_passage-ranking_chinese-tiny')

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

model=NLP_COROM_PASSAGE_RANKING_CHINESE_TINY_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='['功和功率的区别']',arg1='['功反映做功多少，功率反映做功快慢。', '什么是有功功率和无功功率?无功功率有什么用什么是有功功率和无功功率?无功功率有什么用电力系统中的电源是由发电机产生的三相正弦交流电,在交>流电路中,由电源供给负载的电功率有两种;一种是有功功率,一种是无功功率.', '优质解答在物理学中,用电功率表示消耗电能的快慢．电功率用P表示,它的单位是瓦特（Watt）,简称瓦（Wa）符号是W.电流在单位时间内做的功叫做电功率 以灯泡为例,电功率越大,灯泡越亮.灯泡的亮暗由电功率（实际功率）决定,不由通过的电流、电压、电能决定!']')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()