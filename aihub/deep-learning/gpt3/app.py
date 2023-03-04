import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import pysnooper
import os
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics, Trainers
from datasets import Dataset
from modelscope.msdatasets import MsDataset


class GPT3_Model(Model):
    # 模型基础信息定义
    name='gpt3'   # 该名称与目录名必须一样，小写
    label='文本生成模型'
    describe="文本生成模型"
    field="自然语言"
    scenes="机器人问答"
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    train_inputs = [
        Field(Field_type.text, name='file_path', label='文本文件地址', describe='每行一段文字',default='/mnt/'),
        Field(Field_type.text, name='max_epochs', label='最大迭代次数', describe='最大迭代次数', default='10')
    ]

    inference_inputs = [
        Field(type=Field_type.text, name='text', label='文本前段不分',describe='输入前一部分文本，会自动补充后一部分文本',default='今天天气真好，')
    ]
    inference_resource = {
        "resource_gpu": "1"
    }
    # 训练的入口函数，将用户输入参数传递
    # @pysnooper.snoop()
    def train(self,file_path=None,max_epochs=10, **kwargs):
        if not file_path:
            dataset_dict = MsDataset.load('chinese-poetry-collection')
            train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt'})
            eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'})
        else:
            train_dataset=None
            eval_dataset=None

        max_epochs = int(max_epochs)
        tmp_dir = "./gpt3_poetry"

        num_warmup_steps = 100

        def noam_lambda(current_step: int):
            current_step += 1
            return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

        def cfg_modify_fn(cfg):
            cfg.train.lr_scheduler = {
                "type": "LambdaLR",
                "lr_lambda": noam_lambda,
                "options": {"by_epoch": False}
            }
            cfg.train.optimizer = {
                "type": "AdamW",
                "lr": 3e-4
            }
            cfg.train.dataloader = {"batch_size_per_gpu": 16, "workers_per_gpu": 1}
            return cfg

        kwargs = dict(
            model='damo/nlp_gpt3_text-generation_chinese-base',
            train_dataset=train_dataset,
            eval_datase=eval_dataset,
            max_epochs=max_epochs,
            work_dir=tmp_dir,
            cfg_modify_fn=cfg_modify_fn)

        # 构造 trainer 并进行训练
        trainer = build_trainer(
            name=Trainers.nlp_base_trainer, default_args=kwargs)
        trainer.train()
        return tmp_dir+"/output"

    # 加载模型
    def load_model(self,model_dir=None,**kwargs):
        if model_dir:
            from modelscope.models import Model
            self.text_generation_zh = Model.from_pretrained(model_dir)
        else:
            self.text_generation_zh = pipeline(Tasks.text_generation, model='damo/nlp_gpt3_text-generation_chinese-base')

    # 推理
    @pysnooper.snoop()
    def inference(self,text,**kwargs):
        result_zh = self.text_generation_zh(text)
        back=[
            {
                "text":result_zh['text']
            }
        ]
        return back

model=GPT3_Model()

# model.load_model()
# result = model.inference(text='今天天气真好，')  # 测试
# print(result)
if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web
    model.run()
