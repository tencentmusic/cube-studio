import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_MT5_ZERO_SHOT_AUGMENT_CHINESE_BASE_Model(Model):
    # 模型基础信息定义
    name='nlp-mt5-zero-shot-augment-chinese-base'   # 该名称与目录名必须一样，小写
    label='全任务零样本学习-mT5分类增强版-中文-base'
    describe="该模型在mt5模型基础上使用了大量中文数据进行训练，并引入了零样本分类增强的技术，使模型输出稳定性大幅提升。支持任务包含：分类、摘要、翻译、阅读理解、问题生成等等。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "11063"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/nlp_mt5_zero-shot-augment_chinese-base/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='input', label='输入文本',describe='输入文本',default='',validators=Validator(max=1024))
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "text": "文本分类。\\n候选标签：故事,房产,娱乐,文化,游戏,国际,股票,科技,军事,教育。\\n文本内容：他们的故事平静而闪光，一代人奠定沉默的基石，让中国走向繁荣。"
            }
        },
        {
            "label": "示例2",
            "input": {
                "text": "抽取关键词：\\n在分析无线Mesh网路由协议所面临挑战的基础上,结合无线Mesh网络的性能要求,以优化链路状态路由(OLSR)协议为原型,采用跨层设计理论,提出了一种基于链路状态良好程度的路由协议LR-OLSR.该协议引入了认知无线网络中的环境感知推理思想,通过时节点负载、链路投递率和链路可用性等信息进行感知,并以此为依据对链路质量进行推理,获得网络中源节点和目的节点对之间各路径状态良好程度的评价,将其作为路由选择的依据,实现对路由的优化选择,提高网络的吞吐量,达到负载均衡.通过与OLSR及其典型改进协议P-OLSR、SC-OLSR的对比仿真结果表明,LR-OLSB能够提高网络中分组的递交率,降低平均端到端时延,在一定程度上达到负载均衡."
            }
        },
        {
            "label": "示例3",
            "input": {
                "text": "为以下的文本生成标题：\\n在分析无线Mesh网路由协议所面临挑战的基础上,结合无线Mesh网络的性能要求,以优化链路状态路由(OLSR)协议为原型,采用跨层设计理论,提出了一种基于链路状态良好程度的路由协议LR-OLSR.该协议引入了认知无线网络中的环境感知推理思想,通过时节点负载、链路投递率和链路可用性等信息进行感知,并以此为依据对链路质量进行推理,获得网络中源节点和目的节点对之间各路径状态良好程度的评价,将其作为路由选择的依据,实现对路由的优化选择,提高网络的吞吐量,达到负载均衡.通过与OLSR及其典型改进协议P-OLSR、SC-OLSR的对比仿真结果表明,LR-OLSB能够提高网络中分组的递交率,降低平均端到端时延,在一定程度上达到负载均衡."
            }
        },
        {
            "label": "示例4",
            "input": {
                "text": "为下面的文章生成摘要：\\n据统计，今年三季度大中华区共发生58宗IPO交易，融资总额为60亿美元，交易宗数和融资额分别占全球的35%和25%。报告显示，三季度融资额最高的三大证券交易所分别为东京证券交易所、深圳证券交易所和马来西亚证券交易所"
            }
        },
        {
            "label": "示例5",
            "input": {
                "text": "评价对象抽取：\\n颐和园还是挺不错的，作为皇家园林，有山有水，亭台楼阁，古色古香，见证着历史的变迁。"
            }
        },
        {
            "label": "示例6",
            "input": {
                "text": "翻译成英文：\\n如果日本沉没，中国会接收日本难民吗？"
            }
        },
        {
            "label": "示例7",
            "input": {
                "text": "情感分析：\\n外观漂亮，性能不错，屏幕很好。"
            }
        },
        {
            "label": "示例8",
            "input": {
                "text": "根据给定的段落和答案生成对应的问题。\\n段落：跑步后不能马上进食，运动与进食的时间要间隔30分钟以上。看你跑步的量有多大。不管怎么样，跑完步后要慢走一段时间，将呼吸心跳体温调整至正常状态才可进行正常饮食。血液在四肢还没有回流到内脏，不利于消化，加重肠胃的负担。如果口渴可以喝一点少量的水。洗澡的话看你运动量。如果跑步很剧烈，停下来以后，需要让身体恢复正常之后，再洗澡，能达到放松解乏的目的，建议15-20分钟后再洗澡；如果跑步不是很剧烈，只是慢跑，回来之后可以马上洗澡。 \\n 答案：30分钟以上"
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
        
        self.p = pipeline('text2text-generation', 'damo/nlp_mt5_zero-shot-augment_chinese-base')

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

model=NLP_MT5_ZERO_SHOT_AUGMENT_CHINESE_BASE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(text='文本分类。\n候选标签：故事,房产,娱乐,文化,游戏,国际,股票,科技,军事,教育。\n文本内容：他们的故事平静而闪光，一代人奠定沉默的基石，让中国走向繁荣。')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()