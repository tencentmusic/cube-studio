


# VoP: 通用跨模态视频检索模型-系列-bias

#### [**论文 [点击阅读]**](https://arxiv.org/pdf/2211.12764 "论文")

为了对 VoP 进行全方位验证，我们复现了一系列对比方法并纳入VoP的代码架构。这些其他研究者公布的模型也有非常高的使用价值，且可用于独立的 Encoder 页面中以 Leadboard 形式呈现，方便后续研究者follow。本 MODEL CARD 【*VoP: 通用跨模态视频检索模型-系列-bias*】，是对 [Visual Prompt Tuning (ECCV 2022)](https://arxiv.org/pdf/2203.12119.pdf) 在 video-text retrieval task 下的 bias 模型的 **复现** 。

系列工作：

- [VoP: 通用跨模态视频检索模型](https://modelscope.cn/models/damo/cv_vit-b32_retrieval_vop/summary)
- [VoP: 通用跨模态视频检索模型-系列-bias](https://modelscope.cn/models/damo/cv_vit-b32_retrieval_vop_bias/summary)
- [VoP: 通用跨模态视频检索模型-系列-proj](https://modelscope.cn/models/damo/cv_vit-b32_retrieval_vop_proj/summary)
- [VoP: 通用跨模态视频检索模型-系列-partial](https://modelscope.cn/models/damo/cv_vit-b32_retrieval_vop_partial/summary)


VoP是第一个同时具有视频和文字Prompt的端到端视频文本跨模态检索框架，基于Prompt的高效微调与完全微调相比，VoP利用0.1%的训练参数在5个公开的数据集(MSR-VTT-9k, MSR-VTT-7k, DiDeMo, ActivityNet, LSMDC)中获得了1.4%的平均R@1增益，参数开销却减少了6倍。VoP可以实现输入一段自然语言文本做视频特征检索，返回最相关的视频，或是输入一支本地视频做文本特征检索，返回最相关的文本。

利用VoP实现文本（自然语言）直接搜索视频的可视化样例，如下所示：

![可视化展示](description/vis_whitebg.png "vis")

## 期望模型使用方式以及适用范围

### 如何使用

VoP是基于CLIP的快速微调框架，可以适用于任何需要做视频文本跨模态检索的“视频-文本对”数据当中。

#### 代码范例

- 文本搜索视频

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

vop_pipeline = pipeline(Tasks.vop_retrieval, 
                       model='damo/cv_vit-b32_retrieval_vop_bias')

# 输入文本query
input_text = 'a squid is talking'
# 运行pipeline获得结果
result = vop_pipeline(input_text)

print(f'vop output: {result}.')
print('finished!')
```


- 视频搜索文本

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

vop_pipeline = pipeline(Tasks.vop_retrieval, 
                       model='damo/cv_vit-b32_retrieval_vop_bias')

# 输入视频名称
# 如果自定义视频，请放到 'damo/cv_vit-b32_retrieval_vop' 根目录下即可
input_video = 'video10.mp4'
# 运行pipeline获得结果
result = vop_pipeline(input_video)

print(f'vop output: {result}.')
print('finished!')
```


### 模型局限性以及可能的偏差

- 考虑GPU精度、视频解码工具的差异，可能带来一定的性能差异(<0.5%)
- 测试使用的GPU是Tesla T4，显存16127MiB
- 当前版本在python 3.7.9环境测试通过，其他环境下可用性待测试
- 默认基于MSR-VTT-9K数据作为检索底库



## 训练数据介绍

- MSR-VTT 包含10,000个视频，每个视频与大约20个标题配对，我们用MSR-VTT-9k和MSR-VTT-7k来分别指代两种数据分割
- DiDeMo 包含10,000个Flickr视频，有40,000段文本
- ActivityNet 包含20,000个YouTube视频，一个视频的所有描述被串联成一个单一的查询
- LSMDC 包含从202部电影中提取的118,081个视频片段


## 数据评估及结果

VoP在5个公开数据集上的评估结果如下，红色表示相对于基线(全量微调)是负向性能变化，绿色表示正向，"Ours"括号内的是VoP的结果：

![实验结果](description/exp_whitebg.png "exp")


## 相关论文以及引用信息

如果该模型对您有所帮助，请引用下面的相关的论文：

```BibTeX
@inproceedings{Huang2022VoP,
  title     = {VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval},
  author    = {Siteng Huang and Biao Gong and Yulin Pan and Jianwen Jiang and Yiliang Lv and Yuyuan Li and Donglin Wang},
  journal   = {CVPR 2023},
  year      = {2023}
}

@inproceedings{jia2022visual,
  title={Visual prompt tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXIII},
  pages={709--727},
  year={2022},
  organization={Springer}
}
```



