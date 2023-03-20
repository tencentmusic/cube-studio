
# 语音唤醒模型介绍

## 模型描述

&emsp;&emsp;移动端语音唤醒模型，检测关键词为“小云小云”。  
&emsp;&emsp;模型网络结构继承自[论文](https://www.isca-speech.org/archive/interspeech_2018/chen18c_interspeech.html)《Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting》，其主体为4层cFSMN结构(如下图所示)，参数量约750K，适用于移动端设备运行。  
&emsp;&emsp;模型输入采用Fbank特征，训练阶段使用CTC-loss计算损失并更新参数，输出为基于char建模的中文全集token预测，token数共2599个。测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。  
&emsp;&emsp;模型训练采用"basetrain + finetune"的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。由于采用了中文char全量token建模，并使用充分数据进行basetrain，本模型支持基本的唤醒词/命令词自定义功能，但具体性能无法评估。如用户想验证更多命令词，可以通过页面右侧“在线体验”板块自定义设置并录音测试。  
&emsp;&emsp;目前最新ModelScope版本已支持用户在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型。  

<p align="center">
<img src="fig/Illustration_of_cFSMN.png" alt="cFSMN网络框图" width="400" />
<p align="left">

## 使用方式和范围

运行范围：  
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。
- 模型训练需要用户服务器配置GPU卡，CPU训练暂不支持。

使用方式：
- 使用附带的kwsbp工具(Linux-x86_64)直接推理，分别测试正样本及负样本集合，综合选取最优工作点。

使用范围:
- 移动端设备，Android/iOS型号或版本不限，使用环境不限，采集音频为16K单通道。

目标场景:
- 移动端APP用到的关键词检测场景。

### 如何使用

#### 模型推理代码范例：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

kwsbp_16k_pipline = pipeline(
    task=Tasks.keyword_spotting,
    model='damo/speech_charctc_kws_phone-xiaoyun')

kws_result = kwsbp_16k_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav')
print(kws_result)
```

audio_in参数说明：
- 默认传入url地址的小云正样本音频，函数返回单条测试结果。
- 设置本地单条音频路径，如audio_in='LOCAL_PATH'，函数返回单条测试结果。
- 设置本地正样本目录(自动检索该目录下wav格式音频)，如audio_in=['POS_DIR', None]，函数返回全部正样本测试结果。
- 设置本地负样本目录(自动检索该目录下wav格式音频)，如audio_in=[None, 'NEG_DIR']，函数返回全部负样本测试结果。
- 同时设置本地正/负样本目录，如audio_in=['POS_DIR', 'NEG_DIR']，函数返回Det测试结果，用户可保存JSON格式文本方便选取合适工作点。

#### 模型训练代码范例：

&emsp;&emsp;在modelscope-1.2.0及以上版本，我们上线了近场语音唤醒训练功能，并在小云模型库中放置训练所需资源和配置，以及一个迭代好的基线模型。开发者用户可以在此基础上，根据项目需求采集数据以定制自己的唤醒模型，所需训练数据量极少，训练门槛极低。  

环境部署：
- 首先根据文档[环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)新建conda环境并安装Python、深度学习框架以及modelscope语音领域依赖包：  

```sh
$ conda create -n modelscope python=3.7
$ conda activate modelscope
$ pip install torch torchvision torchaudio
$ pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
$ pip install tensorboardX
```

训练流程：
- s1: 手动创建一个本地工作目录，然后配置到work_dir，用于保存所有训练过程产生的文件
- s2: 获取小云模型库中的配置文件，包含训练参数信息，模型ID确保为'damo/speech_charctc_kws_phone-xiaoyun'
- s3: 初始化一个近场唤醒训练器，trainer tag为'speech_kws_fsmn_char_ctc_nearfield'
- s4: 配置准备好的训练数据列表(kaldi风格)，音频列表分为train/cv，标注合为一个文件，然后启动训练。
- s5: 配置唤醒词，多个请使用英文‘,’分隔；配置测试目录和测试数据列表(kaldi风格)，然后启动测试，最终在测试目录生成测试结果文件——score.txt

&emsp;&emsp;训练代码保存文件，如example_kws.py，通过命令行启动训练：  

```sh
$ PYTHONPATH=. torchrun --standalone --nnodes=1 --nproc_per_node=2 example_kws.py
```

&emsp;&emsp;以下是一些训练参考代码：  

```python
# coding = utf-8

import os
from modelscope.utils.hub import read_config
from modelscope.utils.hub import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

def main():
    # s1
    work_dir = './test_kws_training'

    # s2
    model_id = 'damo/speech_charctc_kws_phone-xiaoyun'
    model_dir = snapshot_download(model_id)
    configs = read_config(model_id)
    config_file = os.path.join(work_dir, 'config.json')
    configs.dump(config_file)

    # s3
    kwargs = dict(
        model=model_id,
        work_dir=work_dir,
        cfg_file=config_file,
    )
    trainer = build_trainer(
        Trainers.speech_kws_fsmn_char_ctc_nearfield, default_args=kwargs)

    # s4
    train_scp = './example_kws/train_wav.scp'
    cv_scp = './example_kws/cv_wav.scp'
    trans_file = './example_kws/merge_trans.txt'
    kwargs = dict(
        train_data=train_scp,
        cv_data=cv_scp,
        trans_data=trans_file
    )
    trainer.train(**kwargs)

    # s5
    keywords = '小云小云'
    test_dir = os.path.join(work_dir, 'test_dir')
    test_scp = './example_kws/test_wav.scp'
    trans_file = './example_kws/test_trans.txt'
    rank = int(os.environ['RANK'])
    if rank == 0:
        kwargs = dict(
            test_dir=test_dir,
            test_data=test_scp,
            trans_data=trans_file,
            gpu=0,
            keywords=keywords,
            batch_size=256,
            )
        trainer.evaluate(None, None, **kwargs)

if __name__ == '__main__':
    main()
```

&emsp;&emsp;补充说明：  
- <font size="2">*kaldi列表风格如下所示：*</font>
   - <font size="2">*音频列表为“索引+路径”，中间以Tab分隔。*</font>
   - <font size="2">*标注列表为“索引+标注”，中间以Tab分隔，标注是否分词均可。*</font>
   - <font size="2">*音频与标注的索引顺序无关联，但集合应当一致，训练时会自动丢弃无法同时索引到路径和标注的数据。*</font>
   - <font size="2">*由于我们的建模方式及算法局限，需要中文的训练音频及全内容标注，与训练中文ASR模型相同，开发者需要注意。*</font>
   - <font size="2">*训练数据需包含一定数量对应关键词和非关键词样本，我们建议关键词数据在25小时以上，混合负样本比例在1:2到1:10之间，实际性能与训练数据量、数据质量、场景匹配度、正负样本比例等诸多因素有关，需要具体分析和调整。*</font>

```sh
$ cat wav.scp
kws_pos_example1	/home/admin/data/test/audios/kws_pos_example1.wav
kws_pos_example2	/home/admin/data/test/audios/kws_pos_example2.wav
...
kws_neg_example1	/home/admin/data/test/audios/kws_neg_example1.wav
kws_neg_example2	/home/admin/data/test/audios/kws_neg_example2.wav
...

$ cat trans.txt
kws_pos_example1	小 云 小 云
kws_pos_example2	小 云 小 云
...
kws_neg_example1	帮 我 导航 一下 回 临江 路 一百零八 还要 几个 小时
kws_neg_example2	明天 的 天气 怎么样
...
```

- 训练及测试产生文件如下所示：
   - 训练每一个epoch保存一次checkpoint及对应训练参数，如0.pt和0.yaml。
   - 测试之前根据迭代的cv_loss选取最优的5个(可配置)checkpoint平均参数得到目标模型，如avg_5.pt。
   - 由于我们的唤醒引擎加载kaldi格式文件，所以将目标模型转成了kalid文本格式，如convert.kaldi.txt。
   - 最终部署还需要使用kaldi nnet-copy文件转成bin文件，并打包运用到SDK。如有移动端唤醒部署需求，欢迎搜索并添加钉钉群进行咨询：21295019391。
   - 测试结果保存在$test_dir/score.txt文件中，包含了每一条测试音频是否唤醒以及唤醒后的得分等信息，开发者可以根据该文件进一步统计并画出Det曲线。

```sh
$ tree training_xiaoyunxiaoyun
training_xiaoyunxiaoyun
├── 0.pt
├── 0.yaml
├── ...
├── 9.pt
├── 9.yaml
├── avg_5.pt
├── config.json
├── config.yaml
├── convert.kaldi.txt
├── init.pt
├── tensorboard
│   ├── events.out.tfevents.1673862887.nls-dev-servers011143155239.et2
└── test_dir
    ├── score.txt
    └── stats_小云小云.txt
```

- 我们训练Pipeline的搭建基于github开源项目——[WeKws](https://github.com/wenet-e2e/wekws)，在迁移到ModelScope之前的调研过程中，我们发现并使用了WeKws这一优秀的专注于语音唤醒的工程，它简明的训练脚本、灵活的配置以及良好的代码风格等诸多优点，最终成了我们旧有训练方案的迁移首选，在此对WeKws开发人员及开源社区贡献者致以最诚挚的谢意！


### 模型局限性以及可能的偏差

- 考虑到正负样本测试集覆盖场景不够全面，可能有特定场合/特定人群唤醒率偏低或误唤醒偏高问题。

## 训练数据介绍

- basetrain使用内部移动端ASR数据5000+小时，finetune使用1万条众包安静场景"小云小云"数据以及约20万条移动端ASR数据。


## 模型训练流程

- 模型训练采用"basetrain + finetune"的模式，finetune过程使用目标场景的特定唤醒词数据并混合一定比例的ASR数据。如训练数据与应用场景不匹配，应当针对性做数据模拟。


### 预处理

- finetune模型对众包数据做近场加噪/加速模拟以扩充数据源。


## 数据评估及结果

- 模型在自建9个场景各50句的正样本集（共450条）测试，唤醒率为93.11%；  
在自建的移动端负样本集上测试，误唤醒为40小时0次。

## 相关论文以及引用信息

```BibTeX
@inproceedings{chen18c_interspeech,
  author={Mengzhe Chen and ShiLiang Zhang and Ming Lei and Yong Liu and Haitao Yao and Jie Gao},
  title={{Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting}},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={2663--2667},
  doi={10.21437/Interspeech.2018-1204}
}
```
