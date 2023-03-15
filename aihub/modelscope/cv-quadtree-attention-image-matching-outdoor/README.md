
# 图像匹配算法介绍

## 任务
输入一对图片，图像匹配算法将输出图片间对应像素的位置。

## 模型描述

本模型基于**QuadTree Attention for Vision Transformers**算法，是该算法的官方模型。

技术细节请见：

**QuadTree Attention for Vision Transformers** <br />
Shitao Tang, Jiahui Zhang, Siyu Zhu and Ping Tan
<br />
**ICLR 2022** <br />
**[[Paper](https://arxiv.org/abs/2201.02767)]** |
**[[中文解读](https://zhuanlan.zhihu.com/p/474165095)]**  <br />


<p>
  &emsp;&emsp; <img src="assets/quadtree_match.png" width="600" />
</p>



## 如何使用

### 代码示例(详见tests/pipelines/test_image_matching.py)

```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


task = 'image-matching'
model_id = 'damo/cv_quadtree_attention_image-matching_outdoor'

input_location = [
                    ['https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_matching1.jpg',
                    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_matching2.jpg']
                ]
estimator = pipeline(Tasks.image_matching, model=model_id)
result = estimator(input_location)
kpts0, kpts1, conf = result[0][OutputKeys.MATCHES]
print(f'Found {len(kpts0)} matches')

```


## 模型精度
在ScanNet及MegaDepth上的结果为
| Method           | AUC@5 | AUC@10 | AUC@20 |
|------------------|:----:|:-----:|:------:|
| ScanNet     | 24.9  |  44.7 |  61.8 |
| Megadepth   | 53.5  |  70.2 |  82.2 |

## 更多结果
QuadTreeAttention是通用的transformer build block, 对于图像分类、检测、分割、双目深度估计等任务均适用。

本仓库目前仅包含图像匹配的室外模型（室内数据也可使用本模型），要使用更多模型可见[此处](https://github.com/Tangshitao/QuadTreeAttention)。

## Bibtex

```
@article{tang2022quadtree,
  title={QuadTree Attention for Vision Transformers},
  author={Tang, Shitao and Zhang, Jiahui and Zhu, Siyu and Tan, Ping},
  journal={ICLR},
  year={2022}
}
```