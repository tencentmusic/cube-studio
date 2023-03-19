# OFA-视觉问答(英文)
## News
- 2023年2月:
  - 优化了finetune流程，支持参数更新、自定义数据及脚本分布式训练等，见finetune示例。
- 2022年11月：
  - 新增[OFA Tutorial](https://modelscope.cn/docs/OFA_Tutorial#1.4%20%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83)。

## 视觉问答是什么？
想针对图片提问担心得不到好的答案？OFA模型一定能帮你！你只需要输入任意1张你的图片以及你的问题，**3秒内**就能收获相应的答案。**本页面右侧**提供了在线体验的服务，欢迎使用！

本系列还有如下模型，欢迎试用：
- [large-通用场景-英文](https://modelscope.cn/models/damo/ofa_visual-question-answering_pretrain_large_en/summary)

## 快速玩起来
玩转OFA只需区区以下数行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角`Notebook`按钮，我们为你提供了配备了的环境，CPU和GPU都支持哦，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了!

<p align="center">
    <img src="resources/visual_question_answering.png" alt="demo" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor
model = 'damo/ofa_visual-question-answering_pretrain_huge_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    model_revision='v1.0.1',
    preprocessor=preprocessor)
image = 'https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/visual_question_answering.png'
text = 'what is grown on the plant?'
input = {'image': image, 'text': text}
result = ofa_pipe(input)
print(result[OutputKeys.TEXT]) # ' money'
```


## OFA是什么？
OFA(One-For-All)是通用多模态预训练模型，使用简单的序列到序列的学习框架统一模态（跨模态、视觉、语言等模态）和任务（如图片生成、视觉定位、图片描述、图片分类、文本生成等），详见我们发表于ICML 2022的论文：[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)。

<p align="center">
    <br>
    <img src="resources/OFA_logo_tp_path.svg" width="150" />
    <br>
<p>
<br>

<p align="center">
        <a href="https://github.com/OFA-Sys/OFA">Github</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2202.03052">Paper </a>&nbsp ｜ &nbspBlog
</p>

<p align="center">
    <br>
        <video src="https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/resources/modelscope_web/demo.mp4" loop="loop" autoplay="autoplay" muted width="80%"></video>
    <br>
</p>

### OFA模型规模：

<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Params-en</th><th>Params-zh</th><th>Backbone</th><th>Hidden size</th><th>Intermediate size</th><th>Num. of heads</th><th>Enc layers</th><th>Dec layers</th>
    </tr>
    <tr align="center">
        <td>OFA<sub>Tiny</sub></td><td>33M</td><td>-</td><td>ResNet50</td><td>256</td><td>1024</td><td>4</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Medium</sub></td><td>93M</td><td>-</td><td>ResNet101</td><td>512</td></td><td>2048</td><td>8</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td>180M</td><td>160M</td><td>ResNet101</td><td>768</td></td><td>3072</td><td>12</td><td>6</td><td>6</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td>470M</td><td>440M</td><td>ResNet152</td><td>1024</td></td><td>4096</td><td>16</td><td>12</td><td>12</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Huge</sub></td><td>930M</td><td>-</td><td>ResNet152</td><td>1280</td></td><td>5120</td><td>16</td><td>24</td><td>12</td>
    </tr>
</table>
<br>


## 为什么OFA是视觉问答的最佳选择？
OFA在视觉问答（VQA）任务的VQA 2.0上取得和近期大模型CoCa同等表现（想看榜单，点[这里](https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278)），具体如下：

<table border="1" width="100%">
    <tr align="center">
        <th>Task</th><th colspan="2">VQA</th>
    </tr>
    <tr align="center">
        <td>Split</td><td>test-dev</td><td>test-std</td>
    </tr>
	<tr align="center">
        <td>OFA<sub>Base</sub></td><td>78.0</td><td>78.1</td>
	</tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td>80.4</td><td>80.7</td>
	</tr>
	<tr align="center">
        <td><b>OFA<sub>Huge</sub></b></td><td>82.0</td><td>82.0</td>
	</tr>
</table>
<br>

## 模型训练流程

### 训练数据介绍
本模型训练数据集是预训练数据集。

### 训练流程
模型及finetune细节请参考[OFA Tutorial](https://modelscope.cn/docs/OFA_Tutorial#1.4%20%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83) 1.4节。


### Finetune示例
```python
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.hub import snapshot_download

train_dataset = MsDataset(
    MsDataset.load('vqa_trial', subset_name='vqa_trial', split="train").remap_columns({
        'image': 'image',
        'question': 'text',
        'answer': 'answer'
    }))
test_dataset = MsDataset(
    MsDataset.load('vqa_trial', subset_name='vqa_trial', split="test").remap_columns({
        'image': 'image',
        'question': 'text',
        'answer': 'answer'
    }))


def cfg_modify_fn(cfg):
    cfg.train.hooks = [{
        'type': 'CheckpointHook',
        'interval': 2
    }, {
        'type': 'TextLoggerHook',
        'interval': 1
    }, {
        'type': 'IterTimerHook'
    }]
    cfg.train.dataloader.batch_size_per_gpu = 1
    cfg.train.max_epochs=2
    return cfg


args = dict(
    model='damo/ofa_visual-question-answering_pretrain_huge_en',
    model_revision='v1.0.1',
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    cfg_modify_fn=cfg_modify_fn,
    work_dir = tempfile.TemporaryDirectory().name)
trainer = build_trainer(name=Trainers.ofa, default_args=args)
trainer.train()
```


## 模型局限性以及可能的偏差
为了广泛的的适用性，本OFA模型是预训练ckpt，没有经过VQA 2.0进行finetune，直接在VQA上面进行测试可能出现效果不理想的情况。

我们一直在努力实现更好的模型提供给用户，如有需求欢迎加入modelscope群联系！ 
<br><br><br>


## 相关论文及引用
如果你觉得OFA好用，喜欢我们的工作，欢迎引用：
```
@article{wang2022ofa,
  author    = {Peng Wang and
               An Yang and
               Rui Men and
               Junyang Lin and
               Shuai Bai and
               Zhikang Li and
               Jianxin Ma and
               Chang Zhou and
               Jingren Zhou and
               Hongxia Yang},
  title     = {OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence
               Learning Framework},
  journal   = {CoRR},
  volume    = {abs/2202.03052},
  year      = {2022}
}
```