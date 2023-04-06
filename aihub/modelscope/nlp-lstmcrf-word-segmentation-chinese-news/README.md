
# LSTM通用领域中文分词模型介绍
中文分词任务就是把连续的汉字分隔成具有语言语义学意义的词汇。中文的书写方式不像英文等日耳曼语系语言词与词之前显式的用空格分隔。为了让计算机理解中文文本，通常来说中文信息处理的第一步就是进行文本分词。

中文分词样例:

- 输入: 阿里巴巴集团的使命是让天下没有难做的生意
- 输出: 阿里巴巴/ 集团/ 的/ 使命/ 是/ 让/ 天下/ 没有/ 难/ 做/ 的/ 生意


## 模型描述
本方法采用char-BiLSTM-CRF模型，word-embedding使用[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)。序列标注标签体系(B、I、E、S),四个标签分别表示单字处理单词的起始、中间、终止位置或者该单字独立成词。char-BiLSTM-CRF模型具体结构可以参考论文[Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030.pdf)

## 期望模型使用方式以及适用范围
本模型主要用于给输入中文句子产出分词结果。用户可以自行尝试输入中文句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用chinese-word-segmentation(中文分词)的能力, 默认单句长度不超过默认单句长度不超过126。如需增加单句的切分长度，可以修改[TokenClassificationTransformersPreprocessor](https://github.com/modelscope/modelscope/blob/master/modelscope/preprocessors/nlp/token_classification_preprocessor.py#L223)中的最大sequence长度。

#### 代码范例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor

model_id = 'damo/nlp_lstmcrf_word-segmentation_chinese-news'
model = Model.from_pretrained(model_id)
tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
pipeline_ins = pipeline(task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
result = pipeline_ins(input="今天天气不错，适合出去游玩")
print (result)
# {'output': '今天 天气 不错 ， 适合 出去 游玩'}
```

### 模型局限性以及可能的偏差
本模型基于PKU数据集(通用新闻领域)上训练，在垂类领域中文文本上的分词效果会有降低，请用户自行评测后决定如何使用。

### ## 训练数据介绍
本模型采用新闻领域分词标注数据集PKU标注训练。

## 模型训练流程

### 预处理
PKU数据集标注数据样例:

```
有心 栽 得 梧桐树 ， 自 有 远方 凤凰 来 。
```

数据预处理成(B、I、E、S)标签体系的数据格式, 每一个独立的单字对应一个独立的标签, 预处理后数据样例如下:

```
在 这 辞 旧 迎 新 的 美 好 时 刻 ， 我 祝 大 家 新 年 快 乐 ， 家 庭 幸 福 ！
S-CWS S-CWS B-CWS I-CWS I-CWS E-CWS S-CWS B-CWS E-CWS B-CWS E-CWS S-CWS S-CWS S-CWS B-CWS E-CWS B-CWS E-CWS B-CWS E-CWS S-CWS B-CWS E-CWS B-CWS E-CWS S-CWS
```

### 训练
模型采用1张NVIDIA V100机器训练, 超参设置如下:

```
train_epochs=10
max_sequence_length=256
batch_size=125
learning_rate=5e-5
optimizer=AdamW
```

### 数据评估及结果

模型在PKU测试数据评估结果:

| Model | Precision | Recall | F1    |    Inference speed on CPU   |
|-------|-----------|--------|-------|-------|
|BAStructBERT-Base | 96.44     | 97.31  | 96.87 |  1.0x  |
|BAStructBERT-Lite | 96.66     | 95.59  | 96.12 |  2.91x |
|LSTMCRF| 95.68 | 94.83 | 95.16 | 13.16x |

## 论文引用
char-BiLSTM-CRF模型可以参考下列论文
```BibTex
@inproceedings{lample-etal-2016-neural,
    title = "Neural Architectures for Named Entity Recognition",
    author = "Lample, Guillaume  and
      Ballesteros, Miguel  and
      Subramanian, Sandeep  and
      Kawakami, Kazuya  and
      Dyer, Chris",
    booktitle = "Proceedings of the 2016 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N16-1030",
    doi = "10.18653/v1/N16-1030",
    pages = "260--270",
}
``` 
