import base64
import io,sys,os

import datasets
import pandas as pd
from cubestudio.aihub.model import Model,Field,Field_type,Validator
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import pysnooper
import os
from datasets import Dataset
from modelscope.msdatasets import MsDataset
from torch.utils.tensorboard import SummaryWriter
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers


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
        Field(Field_type.text, name='save_model_dir', label='模型保存目录', describe='模型保存目录，需要配置为分布式存储下的目录',default=''),
        Field(Field_type.text, name='model_type', label='模型类型，续写或问答',describe='模型类型，续写或问答，续写：为自动补全输入，问答：为根据问题返回答案', default='续写',choices=['续写','问答']),
        Field(Field_type.text, name='file_path', label='csv文件地址', describe='每行一段文字，需要csv格式，续写模式，字段名需为src_txt，问答模式，字段名需为src_txt,tgt_txt',default='',validators=Validator(regex='.*csv')),
        Field(Field_type.text, name='max_epochs', label='最大迭代次数', describe='最大迭代次数', default='10')
    ]

    inference_inputs = [
        Field(type=Field_type.text, name='text', label='文本前段不分',describe='输入前一部分文本，会自动补充后一部分文本',default='今天天气真好，')
    ]
    inference_resource = {
        "resource_gpu": "1"
    }

    # 续写训练
    def train_1(self,save_model_dir,file_path=None,max_epochs=10,**kwargs):

        if not file_path:
            dataset_dict = MsDataset.load('chinese-poetry-collection')
            train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt'})
            eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'})
        else:
            data_df = pd.read_csv(file_path)
            train_df = data_df.sample(frac=0.9, random_state=0, axis=0)  # 划分数据集
            eval_df = data_df[~data_df.index.isin(train_df.index)]
            train_dataset = Dataset.from_pandas(train_df)
            eval_dataset = Dataset.from_pandas(eval_df)
            print('训练集', train_dataset.num_rows)
            print('验证集', eval_dataset.num_rows)


        max_epochs = int(max_epochs)
        tmp_dir = save_model_dir

        num_warmup_steps = 100

        def noam_lambda(current_step: int):
            current_step += 1
            return min(current_step ** (-0.5),
                       current_step * num_warmup_steps ** (-1.5))

        def cfg_modify_fn(cfg):
            cfg.train.lr_scheduler = {
                'type': 'LambdaLR',
                'lr_lambda': noam_lambda,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.optimizer = {'type': 'AdamW', 'lr': 3e-4}
            cfg.train.dataloader = {
                'batch_size_per_gpu': 16,
                'workers_per_gpu': 1
            }
            cfg.train.hooks.append({
                'type': 'MegatronHook'
            })
            cfg.evaluation.dataloader = {
                'batch_size_per_gpu': 8,
                'workers_per_gpu': 1
            }
            cfg.evaluation.metrics = 'ppl'
            return cfg

        kwargs = dict(
            model='damo/nlp_gpt3_text-generation_1.3B',
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_epochs=max_epochs,
            work_dir=tmp_dir,
            cfg_modify_fn=cfg_modify_fn)

        # Construct trainer and train
        trainer = build_trainer(
            name=Trainers.gpt3_trainer, default_args=kwargs)
        trainer.train()
        return tmp_dir + "/output"

    # 问答训练
    def train_2(self,save_model_dir,file_path=None,max_epochs=10,**kwargs):
        if not file_path:
            dataset_dict = MsDataset.load('DuReader_robust-QG')
            train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
                .map(lambda example: {'src_txt': example['src_txt'] + '\n'})
            eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
                .map(lambda example: {'src_txt': example['src_txt'] + '\n'})
        else:
            data_df = pd.read_csv(file_path)
            train_df = data_df.sample(frac=0.9, random_state=0, axis=0)  # 划分数据集
            eval_df = data_df[~data_df.index.isin(train_df.index)]
            train_dataset = Dataset.from_pandas(train_df)
            eval_dataset = Dataset.from_pandas(eval_df)
            print('训练集', train_dataset.num_rows)
            print('验证集', eval_dataset.num_rows)

        max_epochs = int(max_epochs)

        tmp_dir = save_model_dir

        num_warmup_steps = 200

        def noam_lambda(current_step: int):
            current_step += 1
            return min(current_step ** (-0.5),
                       current_step * num_warmup_steps ** (-1.5))

        def cfg_modify_fn(cfg):
            cfg.train.lr_scheduler = {
                'type': 'LambdaLR',
                'lr_lambda': noam_lambda,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.train.optimizer = {'type': 'AdamW', 'lr': 1e-4}
            cfg.train.dataloader = {
                'batch_size_per_gpu': 4,
                'workers_per_gpu': 1
            }
            cfg.train.hooks.append({
                'type': 'MegatronHook'
            })
            cfg.preprocessor.sequence_length = 512
            cfg.model.checkpoint_model_parallel_size = 1
            return cfg

        kwargs = dict(
            model='damo/nlp_gpt3_text-generation_1.3B',
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_epochs=max_epochs,
            work_dir=tmp_dir,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(
            name=Trainers.gpt3_trainer, default_args=kwargs)
        trainer.train()
        return tmp_dir + "/output"

    # 训练的入口函数，将用户输入参数传递
    # @pysnooper.snoop()
    def train(self,save_model_dir,model_type='续写',file_path=None,max_epochs=10, **kwargs):
        if model_type=='续写':
            return self.train_1(save_model_dir,file_path,max_epochs,**kwargs)
        if model_type=='问答':
            return self.train_2(save_model_dir,file_path,max_epochs,**kwargs)

    def download_model(self):
        self.text_generation_zh = pipeline(Tasks.text_generation, model='damo/nlp_gpt3_text-generation_1.3B')

    # 加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        if save_model_dir:
            model_path = os.path.join(save_model_dir,'output')
            if os.path.exists(model_path):
                from modelscope.models import Model
                self.text_generation_zh = Model.from_pretrained(save_model_dir)
                return
        self.download_model()

    # 推理
    # @pysnooper.snoop()
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
    # python app.py web --save_model_dir xx
    # python app.py download_model 用于再构建镜像下载一些预训练模型
    model.run()
