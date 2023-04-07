import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_COROM_SENTENCE_EMBEDDING_CHINESE_BASE_ECOM_Model(Model):
    # 模型基础信息定义
    name='nlp-corom-sentence-embedding-chinese-base-ecom'   # 该名称与目录名必须一样，小写
    label='CoROM文本向量-中文-电商领域-base'
    describe="基于CoROM-base预训练语言模型的电商领域中文文本表示模型，基于输入的句子产出对应的文本向量，文本向量可以使用在下游的文本检索、句子相似度计算、文本聚类等任务中。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "10681"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base-ecom/summary"

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
                    "阔腿裤女冬牛仔"
                ],
                "arg1": [
                    "阔腿牛仔裤女秋冬款潮流百搭宽松",
                    "牛仔阔腿裤女大码胖mm高腰显瘦夏季薄款宽松垂感泫雅拖地裤子",
                    "阔腿裤男大码高腰宽松"
                ]
            }
        },
        {
            "label": "示例1",
            "input": {
                "arg0": [
                    "大落地窗"
                ],
                "arg1": [
                    "新豪轩门窗斯图加特78系统铝合金窗内开内倒落地窗断桥铝平开窗",
                    "前景厂家供应电动重型大平移天窗屋顶天井铝合金窗单开对开可定制",
                    "铝合金阳台封闭窄框全景无框折叠窗落地玻璃隔音推拉隐形纱窗定制"
                ]
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": [
                    "鱼珠牌万能胶快干型"
                ],
                "arg1": [
                    "胶粘剂牌胶木木板pc珠环保鱼珠胶胶胶水板强力鱼万能胶工金属皮革",
                    "手办显能胶水专用修复pvc模型粘pp塑料强力万能速干玩具断裂防水",
                    "鱼珠牌胶水粘桌球台布换台尼胶水胶更换台球桌布专用胶水"
                ]
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": [
                    "万仁堂医用半月板跌打损伤筋骨膝盖疼痛专用药膏风湿类关节痛贴膏"
                ],
                "arg1": [
                    "万仁堂医用半月板跌打损伤筋骨膝盖疼痛专用药膏风湿类关节痛贴膏",
                    "天中堂关通舒医用冷敷贴关节腰椎疼痛贴关通舒冷敷贴关通舒贴5贴",
                    "康企医用冷敷敷料痛风关节发炎关节肿胀疼痛",
                    "正品德一堂冷敷凝胶得一堂筋骨冷凝凝胶新堂肩颈冷敷小凝胶包邮"
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
        
        self.p = pipeline('sentence-embedding', 'damo/nlp_corom_sentence-embedding_chinese-base-ecom')

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

model=NLP_COROM_SENTENCE_EMBEDDING_CHINESE_BASE_ECOM_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='['阔腿裤女冬牛仔']',arg1='['阔腿牛仔裤女秋冬款潮流百搭宽松', '牛仔阔腿裤女大码胖mm高腰显瘦夏季薄款宽松垂感泫雅拖地裤子', '阔腿裤男大码高腰宽松']')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()