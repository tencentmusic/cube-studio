import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class PROMPTCLUE_Model(Model):
    # 模型基础信息定义
    name='promptclue'   # 该名称与目录名必须一样，小写
    label='全中文任务支持零样本学习模型'
    describe="支持近20中文任务，并具有零样本学习能力。 针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对生成任务，可以进行采样自由生成。使用1000亿中文token（字词级别）进行大规模预训练，累计学习1.5万亿中文token，并且在100+任务上进行多任务学习获得。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.png'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "27886"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/ClueAI/PromptCLUE/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='input', label='prompt文本',describe='prompt文本',default='',validators=Validator(max=1024))
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "text": "这是关于哪方面的新闻： \\n如果日本沉没，中国会接收日本难民吗？\\n选项：故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏\\n答案："
            }
        },
        {
            "label": "示例2",
            "input": {
                "text": "以下两句话是否表达相同意思：\\n文本1：糖尿病腿麻木怎么办？\\n文本2：糖尿病怎样控制生活方式\\n选项：相似，不相似\\n答案："
            }
        },
        {
            "label": "示例3",
            "input": {
                "text": "阅读以下对话并回答问题。\\n男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。\\n问题：女的怎么样？\\n选项：正在工作，感冒了，在打电话，要出差。\\n答案："
            }
        },
        {
            "label": "示例4",
            "input": {
                "text": "信息抽取：\\n张玄武1990年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。\\n问题：机构，人名，职位，籍贯，专业，国籍，种族\\n答案："
            }
        },
        {
            "label": "示例5",
            "input": {
                "text": "抽取关键词：\\n当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。\\n关键词："
            }
        },
        {
            "label": "示例6",
            "input": {
                "text": "翻译成中文：\\nThis is a dialogue robot that can talk to people.\\n答案："
            }
        },
        {
            "label": "示例7",
            "input": {
                "text": "为下面的文章生成摘要：\\n北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡\\n摘要："
            }
        },
        {
            "label": "示例8",
            "input": {
                "text": "推理关系判断：\\n前提：小明明天要去北京\\n假设：小明计划明天去上海\\n选项：矛盾，蕴含，中立\\n答案："
            }
        },
        {
            "label": "示例9",
            "input": {
                "text": "问答：\\n问题：小米的创始人是谁？\\n答案："
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
        
        self.p = pipeline('text2text-generation', 'ClueAI/PromptCLUE')

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

model=PROMPTCLUE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(text='这是关于哪方面的新闻： \n如果日本沉没，中国会接收日本难民吗？\n选项：故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏\n答案：')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()