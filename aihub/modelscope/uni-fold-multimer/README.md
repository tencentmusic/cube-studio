
# Uni-Fold-Multimer 开源的蛋白质复合物结构预测模型

## 模型描述
输入蛋白质多聚体的一级结构（1D序列），预测蛋白质的三级结构（3D位置），同时给出预测结果的置信度。

## 期望模型使用方式以及适用范围
本模型主要用于蛋白质复合的预测。

### 如何使用
目前仅支持GPU运行，需要额外安装[Uni-Core](https://github.com/dptech-corp/Uni-Core/releases)。

#### 代码范例
用空格隔开蛋白质的多条链。
```python
from modelscope.pipelines import pipeline

pipeline_ins = pipeline(
    task='protein-structure',
    model='DPTech/uni-fold-multimer')
protein = 'GSSSQVQLVESGGGLVQAGGSLRLS GSSSQVQLVESGGGLVQAGGSLRLS GSSSQVQLVESGGGLVQAGGSLRLS'
outputs = pipeline_ins(protein)
```

### 模型局限性以及可能的偏差
模型性能依赖于同源序列的数量，如果同源序列数量少，模型可能无法保证预测结果。

对于超大多聚体，预测时间需要较久，显存也可能不够。

## 训练代码及训练数据
模型根据AlphaFold的论文描述训练，训练代码开源在[GitHub](https://github.com/dptech-corp/Uni-Fold)。

训练使用的数据 [Uni-Fold-Data](https://modelscope.cn/datasets/DPTech/Uni-Fold-Data/summary) 也在 ModelScope 上开放。


### 相关论文以及引用信息
```bibtex
@article {uni-fold,
	author = {Li, Ziyao and Liu, Xuyang and Chen, Weijie and Shen, Fan and Bi, Hangrui and Ke, Guolin and Zhang, Linfeng},
	title = {Uni-Fold: An Open-Source Platform for Developing Protein Folding Models beyond AlphaFold},
	year = {2022},
	doi = {10.1101/2022.08.04.502811},
	URL = {https://www.biorxiv.org/content/10.1101/2022.08.04.502811v3},
	eprint = {https://www.biorxiv.org/content/10.1101/2022.08.04.502811v3.full.pdf},
	journal = {bioRxiv}
}
```
