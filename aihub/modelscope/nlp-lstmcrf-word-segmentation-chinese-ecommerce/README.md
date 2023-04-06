
# LSTM电商领域中文分词模型介绍
中文分词任务就是把连续的汉字分隔成具有语言语义学意义的词汇。中文的书写方式不像英文等日耳曼语系语言词与词之前显式的用空格分隔。为了让计算机理解中文文本，通常来说中文信息处理的第一步就是进行文本分词。

## 模型描述
本方法采用char-BiLSTM-CRF模型，word-embedding使用[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors), 序列标注标签体系(B、I、E、S),四个标签分别表示单字处理单词的起始、中间、终止位置或者该单字独立成词。char-BiLSTM-CRF模型具体结构可以参考论文[Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030.pdf)

电商领域的分词训练数据基于电商搜索Query和标题数据标注得到, 对比通用领域分词模型, 主要提升对电商领域特有的品牌、品类、商品修饰等词汇的切分准确率

```
- 输入: cos风修身吊带针织连衣裙女收腰显瘦小黑裙长裙
- 通用领域分词结果: cos 风 修身 吊带 针织 连衣裙 女 收 腰 显 瘦 小 黑裙 长裙
- 电商领域分词结果: cos风 修身 吊带 针织 连衣裙 女 收腰 显瘦 小黑裙 长裙
```

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

model_id = 'damo/nlp_lstmcrf_word-segmentation_chinese-ecommerce'
model = Model.from_pretrained(model_id)
tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
pipeline_ins = pipeline(task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
result = pipeline_ins(input="收腰显瘦黑裙长裙")
print (result)
# {'output': '收腰 显瘦 黑裙 长裙'}
```

### 模型局限性以及可能的偏差
本模型基于电商领域分词数据训练，在其它领域中文文本上的分词效果会有降低，请用户自行评测后决定如何使用。

### 训练
模型采用1张NVIDIA V100机器训练, 超参设置如下:

```
train_epochs=10
max_sequence_length=256
batch_size=64
learning_rate=5e-5
optimizer=AdamW
```

### 数据评估及结果

模型在电商标题、Query测试数据评估结果:

| Model | Precision | Recall | F1    |
|-------|-----------|--------|-------|
|BAStructBERT-Base | 97.89     | 98.20  | 98.04 |
|LSTMCRF | 96.88 | 97.02 | 96.94 |

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
