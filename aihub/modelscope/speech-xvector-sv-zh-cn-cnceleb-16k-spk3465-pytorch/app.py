import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class SPEECH_XVECTOR_SV_ZH_CN_CNCELEB_16K_SPK3465_PYTORCH_Model(Model):
    # 模型基础信息定义
    name='speech-xvector-sv-zh-cn-cnceleb-16k-spk3465-pytorch'   # 该名称与目录名必须一样，小写
    label='xvector说话人确认-中文-cnceleb-16k-离线-pytorch'
    describe="该模型是使用CN-Celeb 1&2以及AliMeeting数据集预训练得到的说话人嵌入码（speaker embedding）提取模型。可以直接用于通用和会议场景的说话人确认和说话人日志等任务。在CN-Celeb语音测试集上EER为9.00%，在AliMeeting测试集上的EER为1.45%。"
    field="听觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "998"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.audio, name='enroll', label='说话人音频',describe='如果仅有此输入，则输出embedding',default='',validators=None),
        Field(type=Field_type.audio, name='input', label='说话人音频',describe='如果同时包含此输入，则输出两音频是同一个人的概率',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "enroll": "/mnt/workspace/.cache/modelscope/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/example/sv_example_enroll.wav",
                "input": "/mnt/workspace/.cache/modelscope/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/example/sv_example_different.wav"
            }
        },
        {
            "label": "示例2",
            "input": {
                "enroll": "/mnt/workspace/.cache/modelscope/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/example/sv_example_enroll.wav",
                "input": "/mnt/workspace/.cache/modelscope/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/example/sv_example_same.wav"
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
        
        self.p = pipeline('speaker-verification', 'damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch')

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,enroll,input=None,**kwargs):
        if input:
            result = self.p(audio_in=(enroll,input))
            back = [
                {
                    "text": str(result["test1"])+"%"
                }
            ]
        else:
            file_name = os.path.basename(enroll)
            result = self.p(audio_in=enroll)
            back = [
                {
                    "text": str(result[file_name].tolist())   # 获取声纹数组
                }
            ]
        return back

model=SPEECH_XVECTOR_SV_ZH_CN_CNCELEB_16K_SPK3465_PYTORCH_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# # result = model.inference(enroll='/mnt/workspace/.cache/modelscope/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/example/sv_example_enroll.wav',input='/mnt/workspace/.cache/modelscope/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/example/sv_example_same.wav')  # 测试
# result = model.inference(enroll='/mnt/workspace/.cache/modelscope/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/example/sv_example_enroll.wav')  # 测试
#
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    model.run()

# 模型大小 72M
# 两个人声计算相似度 v100 gpu上  0.3s 占用显存2G
# 一个人声计算相似度 v100 gpu上  0.3s 占用显存2G