# OFA-视觉定位(英文)
## 视觉定位是什么？
如果你想找出某个物体在图片上的位置，你只需要输入对这个物体的描述，比如“a blue turtle-like pokemon with round head”， OFA模型便能框出它的所在位置。**本页面右侧**提供了在线体验的服务，欢迎使用！

本系还有如下模型，欢迎试用：
- [large-通用场景-中文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_large_zh/summary)
- [tiny-通用场景-英文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_distilled_en/summary)

## 快速玩起来
玩转OFA只需区区以下数行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角`Notebook`按钮，我们为你提供了配备好的环境（可选CPU/GPU），你只需要在notebook里输入提供的代码，就可以把OFA玩起来了！
<p align="center">
    <img src="resources/visual_grounding.png" alt="皮卡丘" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_large_en')
image = 'https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-grounding/visual_grounding.png'
text = 'a blue turtle-like pokemon with round head'
input = {'image': image, 'text': text}
result = ofa_pipe(input)
print(result[OutputKeys.BOXES])
```
<br>

## OFA是什么？
OFA(One-For-All)是通用多模态预训练模型，使用简单的序列到序列的学习框架统一模态（跨模态、视觉、语言等模态）和任务（如图片生成、视觉定位、图片描述、图片分类、文本生成等），详见我们发表于ICML 2022的论文：[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)，以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)。
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



## 为什么OFA是视觉定位的最佳选择？
OFA在视觉定位任务的经典公开数据集RefCOCO、RefCOCO+、RefCOCOg均取得当前最优表现，具体结果如下：

<table border="1" width="100%">
    <tr align="center">
        <th>Task</th><th>RefCOCO</th><th>RefCOCO+</th><th>RefCOCOg</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td colspan="3">Acc@0.5</td>
    </tr>
	<tr align="center">
        <td>Split</td></td><td>val / test-a / test-b</td><td>val / test-a / test-b</td><td>val-u / test-u</td>
    </tr>
	<tr align="center">
        <td>OFA<sub>Base</sub></td><td>88.48 / 90.67 / 83.30</td><td>81.39 / 87.15 / 74.29</td><td>82.29 / 82.31</td>
	</tr>
    <tr align="center">
        <td><b>OFA<sub>Large</sub></td><td>90.05 / 92.93 / 85.26</td><td>85.80 / 89.87 / 79.22</td><td>85.89 / 86.55</td>
	</tr>
	<tr align="center">
        <td>OFA<sub>Huge</sub></td><td>92.04 / 94.03 / 88.44</td><td>87.86 / 91.70 / 80.71</td><td>88.07 / 88.78</td>
	</tr>
</table>
<br>

## 模型训练流程

### 训练数据介绍
本模型训练数据集是refcoco数据集。

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
```