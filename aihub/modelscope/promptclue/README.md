
# 全中文任务支持零样本学习模型

<a href='https://github.com/clue-ai/PromptCLUE'>PromptCLUE</a>：支持最多中文任务的开源预训练模型

这个模型是基于1000亿token中文语料上预训练，并且在数百种任务上进行Prompt任务式训练。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对多种生成任务，可以进行采样自由生成。 

任务简要描述：

  1. 分类任务：输入提示、文本和分类选项，输出文本所属的种类；

  2.自然语言推理任务：输入提示、两段文本，输出两者所属关系；

  3.阅读理解任务：输入提示、参考文本和问题（以及选项），输出问题的答案；

  4.生成任务：输入提示、文本和问题，输出按照要求生成的文本（详细示例见代码范例）

<a href='https://www.cluebenchmarks.com/clueai.html' targe='_blank'>在线demo</a> | <a href='https://huggingface.co/ClueAI/PromptCLUE' targe='_blank'>huggingface下载地址</a> |   <a href='https://colab.research.google.com/drive/1noyBA_JrYO6Lk6cwxsNZ_jdJ-Jtaf82G?usp=sharing#scrollTo=Nk2tSi3vnSN0' targe='_blank'>colab使用示例</a> |  <a href='https://colab.research.google.com/drive/1QIQDWAACkV7-iRrkrk18XrRjEekMhOtv?usp=sharing' targe='_blank'>自定义数据集进行训练</a> |  <a href='https://github.com/CLUEbenchmark/pCLUE' targe='_blank'>prompt中文数据集</a>

## 模型描述

支持几十个不同类型的任务，具有较好的零样本学习能力和少样本学习能力。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对生成任务，可以进行采样自由生成。千亿中文token上大规模预训练，累计学习1.5万亿中文token，亿级中文任务数据上完成训练，训练任务超过150+。比base版平均任务提升7个点+；具有更好的理解、生成和抽取能力，并且支持文本改写、纠错、知识图谱问答。
实现了中文上的三大统一：统一模型框架，统一任务形式，统一应用方式。
- 统一模型框架：采用Text-to-Text的生成式预训练模型进行统一建模。
- 统一任务形式：Prompt统一不同的NLP任务间的差异，转化为统一的text-to-text数据形式。
- 统一应用方式：对目标任务形成拿来即用的模型，下游应用时都可转化为统一的prompt自适应方式，进行zero-shot/few-shot测试。

<div align=center><img src="./prompt.png"/></div>

## 期望模型使用方式及适用范围

### 如何使用

在安装完成ModelScope之后即可使用PromptCLUE的能力，解决全中文NLP问题


#### 代码范例

加载模型：
```python
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models.nlp import T5ForConditionalGeneration
    from modelscope.preprocessors import TextGenerationT5Preprocessor
```
使用模型进行预测推理方法：
```python
    model = T5ForConditionalGeneration.from_pretrained('ClueAI/PromptCLUE', revision='v0.1')
    preprocessor = TextGenerationT5Preprocessor(model.model_dir)
    pipeline_t2t = pipeline(task=Tasks.text2text_generation, model=model, preprocessor=preprocessor)

    print(pipeline_t2t('情感分析：\n这个看上去还可以，但其实我不喜欢\n选项：积极，消极'))
    # {'text': '消极'}

    print(pipeline_t2t("下面句子是否表示了相同的语义：\n文本1：糖尿病腿麻木怎么办？\n文本2：糖尿病怎样控制生活方式\n选项：相似，不相似\n答案："))
    # {'text': '不相似'}

    print(pipeline_t2t('这是关于哪方面的新闻：\n如果日本沉没，中国会接收日本难民吗？\n选项：故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏'))
    # {'text': '国际'}
    
    print(pipeline_t2t("阅读文本抽取关键信息：\n张玄武1990年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。\n问题：机构，人名，职位，籍贯，专业，国籍，学历，种族\n答案："))
    # {'text': '机构：杭州线锁科技技术_人名：张玄武_职位：博士学历'}

    print(pipeline_t2t("翻译成英文：\n杀不死我的只会让我更强大\n答案："))
    # {'text': 'To kill my life only let me stronger'}

    print(pipeline_t2t('为下面的文章生成摘要：\n北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡'))
    # {'text': '四川甘孜发生6.8级地震'}
    
    print(pipeline_t2t("推理关系判断：\n前提：小明今天在北京\n假设：小明在深圳旅游\n选项：矛盾，蕴含，中立\n答案："))
    # {'text': '蕴涵'}

    print(pipeline_t2t('阅读以下对话并回答问题。\n男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。\n问题：女的怎么样？\n选项：正在工作，感冒了，在打电话，要出差。'))
    # {'text': '感冒了'}

    print(pipeline_t2t("文本纠错：\n告诉二营长，叫他彻回来，我李云龙从不打没有准备的杖\n答案："))
    #{'text'：'告诉二营长，叫他下来，我李云龙从不打没有准备的仗'}

    print(pipeline_t2t("问答：\n问题：小米的创始人是谁？\n答案："))
    # {'text': '小米创始人：雷军'}
```


### 模型局限性及可能的偏差

我们的模型基于大规模NLP数据集（如<a href='https://github.com/CLUEbenchmark/pCLUE'>pCLUE</a>），各领域综合表现素质较高，但在某些垂直领域可能表现稍弱；

## 训练数据介绍

pCLUE：基于提示的大规模预训练数据集，用于多任务学习和零样本学习

### 目前已经有包含9个数据集：

    1.单分类tnews 
    2.单分类iflytek 
    3.自然语言推理ocnli 
    4.语义匹配afqmc 
    5.指代消解-cluewsc2020 
    6.关键词识别-csl 
    7.阅读理解-自由式c3 
    8.阅读理解-抽取式cmrc2018 
    9.阅读理解-成语填空chid 
    
### 字段说明及评价标准：
    input:模型的输入
    target:模型的输出
    type:任务类型，阅读理解(mrc),分类(classify)，生成(generate)，自然语言推理(nli)
    评价标准：阅读理解(em),分类(acc)，生成(em)，自然语言推理(acc)
    answer_choices:选项（只有分类、推理类任务有）

## 数据评估及结果

效果对比--16类中文任务

|  任务类型  | PromptCLUE-base  | PromptCLUE-large    | 
| :----:| :----: | :----: | 
|  **分数** Score  | 63.47  | 70.55(+7.08)   | 
|   参数 Parameters  | 220M |  770M   |  
| **理解任务**（acc，10类） |  | | 
| 分类 classify | 89.56 | 92.89| 
| 情感分析 emotion_analysis | 80.55 | 85.64 | 
| 相似度计算 similar | 70.94 | 78.47 | 
| 自然语言推理 nli | 78.00 | 86.67 | 
| 指代消解 anaphora_resolution | 30.00 | 64.00| 
| 阅读理解 reading_comprehension | 71.69 | 84.78 | 
| 关键词提取 keywords_extraction | 41.44 | 47.78 | 
| 信息抽取 ner | 63.02 | 70.09 | 
| 知识图谱问答 knowledge_graph  | - | 53.11 |
| 中心词提取 Keyword_extraction | 66.50 |71.50 |  
| **生成任务**（rouge，6类） |  |   | 
| 翻译（英中、中英） nmt | 55.92 | 59.67 | 
| 摘要 summary | 31.71 | 34.48| 
| 问答 qa | 21.18 | 27.05 | 
| 生成（文章、问题生成） | 35.86 | 39.87 | 
| 改写 paraphrase | - | 57.68  | 
| 纠错 correct | - | 93.35  | 


