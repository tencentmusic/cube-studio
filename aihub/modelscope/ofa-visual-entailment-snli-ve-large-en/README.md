# OFA-图文蕴含 (英文)
## 图文蕴含是什么？
图文蕴含即根据给定的图片和文本判断其语义关系，从“entailment”、“contradiction”和“neutrality”三种关系中选出。

本系列还有如下模型，欢迎试用：
- [tiny-通用场景-英文](https://modelscope.cn/models/damo/ofa_visual-entailment_snli-ve_distilled_v2_en/summary)

## 快速玩起来
玩转OFA只需区区以下数行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角`Notebook`按钮，我们为你提供了配备了GPU的环境，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了！

<p align="center">
    <img src="resources/visual_entailment.jpg" alt="dogs" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
ofa_pipe = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')
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


## 为什么OFA是图文蕴含的最佳选择？
OFA在图文蕴含的数据集SNLI-VE上取得最优表现，具体结果如下：

<table border="1" width="100%">
    <tr align="center">
        <th>Task</th><th colspan="2">SNLI-VE</th>
    </tr>
    <tr align="center">
        <td>Split</td><td>test-dev</td><td>test-std</td>
    </tr>
	<tr align="center">
        <td>OFA<sub>Base</sub></td><td>89.3</td><td>89.2</td>
	</tr>
    <tr align="center">
        <td><b>OFA<sub>Large</sub></b></td><td>90.3</td><td>90.2</td>
	</tr>
	<tr align="center">
        <td>OFA<sub>Huge</sub></td><td>91.0</td><td>91.2</td>
	</tr>
</table>
<br><br>

## 模型训练流程

### 训练数据介绍
本模型训练数据集是snli-ve数据集。

### 训练流程
finetune能力请参考[OFA Tutorial](https://modelscope.cn/docs/OFA_Tutorial#1.4%20%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83) 1.4节。

## 模型局限性以及可能的偏差
训练数据集自身有局限，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 相关论文以及引用信息
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