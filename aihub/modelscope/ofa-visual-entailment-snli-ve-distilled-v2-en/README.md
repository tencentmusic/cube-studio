# OFA-图文蕴含 (英文)
## 图文蕴含是什么？
图文蕴含即根据给定的图片和文本判断其语义关系，从“entailment”、“contradiction”和“neutrality”三种关系中选出。

本系列还有如下模型，欢迎试用：
- [large-通用场景-英文](https://modelscope.cn/models/damo/ofa_visual-entailment_snli-ve_large_en/summary)

## 快速玩起来
玩转OFA只需区区以下数行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角`Notebook`按钮，我们为你提供了配备了GPU的环境，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了！

<p align="center">
    <img src="https://modelscope.cn/api/v1/models/damo/ofa_visual-entailment_snli-ve_distilled_v2_en/repo?Revision=master&FilePath=resources/visual_entailment.jpg&View=true" alt="dogs" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

ofa_pipe = pipeline(
    Tasks.visual_entailment,
    model='damo/ofa_visual-entailment_snli-ve_distilled_v2_en')
image = 'https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-entailment/visual_entailment.jpg'
text = 'there are two birds.'
input = {'image': image, 'text': text}
result = ofa_pipe(input)
print(result[OutputKeys.LABELS]) # no
```
<br>

## OFA是什么？
OFA(One-For-All)是通用多模态预训练模型，使用简单的序列到序列的学习框架统一模态（跨模态、视觉、语言等模态）和任务（如图片生成、视觉定位、图片描述、图片分类、文本生成等），详见我们发表于ICML 2022的论文：[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)。
<br>

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

## OFA的轻量化

本模型是在对OFA的large模型版本基础上通过知识蒸馏进行轻量化而得到的tiny版本模型（参数量33M），方便用户部署在存储和计算资源受限的设备上。蒸馏框架介绍，详见论文：[Knowledge Distillation of Transformer-based Language Models Revisited](https://arxiv.org/abs/2206.14366)，以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA-Compress](https://github.com/OFA-Sys/OFA-Compress)

![distill](./resources/distill_framework.png)

模型效果如下：

![ofa-image-caption](./resources/ve_demo.png)

具体指标如下：

<table>
<thead>
  <tr align="center">
    <th>Task</th>
    <th colspan="2">SNLI-VE</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <td>Split</td>
    <td>val</td>
    <td>test</td>
  </tr>
  <tr align="center">
    <td>OFA-tiny<br>(直接finetune得到)</td>
    <td>85.3</td>
    <td>85.2</td>
  </tr>
  <tr align="center">
    <td>OFA-distill-tiny<br>(通过蒸馏得到)</td>
    <td>87.0</td>
    <td>86.9</td>
  </tr>
</tbody>
</table>

## 模型训练流程

### 训练数据介绍
本模型训练数据集是snli-ve数据集。

### 训练流程
开发中，敬请等待。

## 模型局限性以及可能的偏差
训练数据集自身有局限，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 相关论文以及引用信息
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```
@article{Lu2022KnowledgeDO,
  author    = {Chengqiang Lu and 
               Jianwei Zhang and 
               Yunfei Chu and 
               Zhengyu Chen and 
               Jingren Zhou and 
               Fei Wu and 
               Haiqing Chen and 
               Hongxia Yang},
  title     = {Knowledge Distillation of Transformer-based Language Models Revisited},
  journal   = {ArXiv},
  volume    = {abs/2206.14366}
  year      = {2022}
}
```
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