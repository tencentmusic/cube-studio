# OFA-文本生成图像 (英文)
## 文本生成图像是什么？
文本生成图像即根据输入的文本，生成与文本描述一致的图像。在本模型中，OFA将根据给定文本输出分辨率为256*256的图片。

注1：OFA实现文本生成图像这个任务是在以One For All的理念进行的学术探索，使用的是seq2seq的思路，通过标准自回归的方式生成图像，和目前学术界比较流行的diffusion model不一样。

注2：本模型采样得到的图片将基于CLIP计算的图文相似度进行排序，返回与文本最相符的一张图片。通常采样数量越多，返回结果质量越高，但同时生成过程的显存占用也更大，可以通过增大beam_size的方式增大采样数量，建议设置16及以上。

<br><br>

## 效果展示

注：此处效果图为采样数量（beam_size)为24时得到

![case](resources/case1.png)
<br><br>


## 快速玩起来
玩转OFA只需区区以下数行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角`Notebook`按钮，我们为你提供了配备了GPU的环境，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了！

注：目前我们测试EAIS环境有些问题，还请您使用DSW资源，如下图所示：

<img src="resources/dsw.png" width="400" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.outputs import OutputKeys

model = 'damo/ofa_text-to-image-synthesis_coco_large_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                    model=model, preprocessor=preprocessor)

# 可以通过修改generator的beam_size的方式扩大采样数，理论上采样数量越多效果越好，但显存占用也越大
# 模型采样数建议设置16及以上，notebook内(16G-V100)使用最大仅可设置为4，效果可能不理想
ofa_pipe.model.generator.beam_size=16
# 可以通过修改generator的temperature的方式调整生成质量，temperature越大生成纹理细节越多，相应地，生成结果有更大的可能会有形变
# 建议调整范围为0.8-1.3
ofa_pipe.model.generator.temperature=1.2
result = ofa_pipe({"text":'A photo of a golden palace in the middle of a lake, digital art, HD.'})
result[OutputKeys.OUTPUT_IMGS][0].save('result.png')
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

## 模型训练流程

### 训练数据介绍
本模型训练数据集是coco caption数据集。

### 训练流程
开发中，敬请等待。

## 模型局限性以及可能的偏差
训练数据集自身有局限，有可能产生一些偏差，请用户自行评测后决定如何使用。
模型生成存在随机性性，采样数量（beam size）较小时生成结果可能不理想，建议在显存较大的gpu上将beam_size设置为16及以上。

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