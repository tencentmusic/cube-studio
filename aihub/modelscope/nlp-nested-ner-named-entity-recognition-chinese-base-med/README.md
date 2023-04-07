
# NestedNER介绍

## 模型描述
本方法采用Global Pointer模型，使用nezha-cn-base作为预训练模型底座。模型训练由[AdaSeq](https://github.com/modelscope/AdaSeq)框架支持。
模型结构如下图所示：

![模型结构](description/global_pointer_model.jpg)

可参考论文：[Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition](https://arxiv.org/abs/2208.03054)


## 期望模型使用方式以及适用范围
本模型主要用于给输入中文句子产出命名实体识别结果。用户可以自行尝试输入中文句子。具体调用方式请参考代码示例。

### 如何使用
本模型除安装ModelScope外，还需单独安装AdaSeq，安装方法：
```
pip install adaseq
```
之后即可使用AdaSeq中的模型库实现named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_nested-ner_named-entity-recognition_chinese-base-med')
result = ner_pipeline('1、可测量目标： 1周内胸闷缓解。2、下一步诊疗措施：1.心内科护理常规，一级护理，低盐低脂饮食，留陪客。2.予“阿司匹林肠溶片”抗血小板聚集，“呋塞米、螺内酯”利尿减轻心前负荷，“瑞舒伐他汀”调脂稳定斑块，“厄贝沙坦片”降血压抗心机重构')

print(result)
```

### 模型局限性以及可能的偏差
本模型基于内部医疗数据集alimed上训练，在垂类领域中文文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- alimed: 内部中文医疗命名实体识别数据集，共5551个句子。

| 英文名          | 实体类型 |
|-----------------|--------|
| BODY            | 部位 |
| DIAGNOSIS       | 疾病 |
| DRUG_DOSAGE     | 药品剂量 |
| DRUG_DURATION   | 药品持续时间 |
| DRUG_FORM       | 药品形式 |
| DRUG_FREQUENCY  | 药品频率 |
| DRUG_NAME       | 药品名 |
| DRUG_ROUTE      | 给药方式 |
| DRUG_STRENGTH   | 药品强度 |
| INSTRUMENT      | 医疗器械 |
| OPERATION       | 手术 |
| OTHER_TREATMENT | 其他治疗 |
| SYMPTOM         | 症状体征 |
| TEST_NAME       | 检查名 |
| TEST_RESULT     | 检查结果 |

## 数据评估及结果
模型在alimed测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| alimed | 73.57 | 59.74 | 65.93 |

各个类型的性能如下: 
| Dataset         | Precision | Recall | F1     |
| --------------- | --------- | ------ | ------ |
| BODY            | 72.773    | 59.168 | 65.269 |
| DIAGNOSIS       | 68.371    | 76.08  | 72.02  |
| DRUG_DOSAGE     | 60.87     | 15.73  | 25.0   |
| DRUG_DURATION   | 36.364    | 66.667 | 47.059 |
| DRUG_FORM       | 86.192    | 88.412 | 87.288 |
| DRUG_FREQUENCY  | 73.381    | 88.696 | 80.315 |
| DRUG_NAME       | 87.703    | 92.195 | 89.893 |
| DRUG_ROUTE      | 83.721    | 87.805 | 85.714 |
| DRUG_STRENGTH   | 54.321    | 77.876 | 64.0   |
| INSTRUMENT      | 23.529    | 15.385 | 18.605 |
| OPERATION       | 56.835    | 54.483 | 55.634 |
| OTHER_TREATMENT | 63.784    | 38.816 | 48.262 |
| SYMPTOM         | 69.416    | 63.316 | 66.226 |
| TEST_NAME       | 78.638    | 57.216 | 66.238 |
| TEST_RESULT     | 82.126    | 45.871 | 58.864 |


### 相关论文以及引用信息
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@article{Su2022GlobalPN,
  title={Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition},
  author={Jianlin Su and Ahmed Murtadha and Shengfeng Pan and Jing Hou and Jun Sun and Wanwei Huang and Bo Wen and Yunfeng Liu},
  journal={ArXiv},
  year={2022},
  volume={abs/2208.03054}
}
```
