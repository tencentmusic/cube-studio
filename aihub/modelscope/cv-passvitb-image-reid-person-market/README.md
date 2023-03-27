
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# passvitb-image-reid-person 模型介绍
本模型为图像特征表示提取别模型，使用ViT作为主干网络，输入图像，输出图像的特征表示（image embedding），图像的特征表示可以用于计算两张图片之间的相似程度，从而判断两张图片中的人是不是同一个个体。

## 模型描述
该模型以ViT作为主干网络，使用PASS方法进行自监督预训练，并在下游数据集上finetune。之前的重识别自监督工作，已经证明了在无标签的图片数据上预训练的效果优于直接使用通用分类的预训练模型，但是之前的重识别自监督工作并没有对自监督方法做针对性的改进，而本模型提出的PASS方法结合重识别任务的特点，将人体局部特征引入到自监督的过程中，更加适合重识别任务，最终也达到更好的效果。相应的论文发表于ECCV 2022，在多个数据集上达到SOTA。PASS的整体流程如下图所示：

![PASS流程图](https://modelscope.cn/api/v1/models/damo/cv_passvitb_image-reid-person_market/repo?Revision=master&FilePath=assets/PASS_overview.jpg&View=true)

## 期望模型使用方式以及适用范围
该模型适用于行人重识别场景，输入包含人的图像，输出图像的特征表示，可利用该特征表示进行后续的相似度计算和图像排序。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型，得到图像的特征表示，本模型的输出为一个1536维的特征向量。

#### 代码范例
基础示例代码。下面的示例代码展示的是如何通过一张图片作为输入，得到图片对应的特征向量。
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_passvitb_image-reid-person_market'
input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_reid_person.jpg'

image_reid_person = pipeline(Tasks.image_reid_person, model=model_id)
result = image_reid_person(input_location)
print("result is : ", result[OutputKeys.IMG_EMBEDDING])
```

相似度示例代码。在实际使用过程中，得到图片的特征向量通常不是最终的期望的输出形式，用户可以基于特征向量进行图片之间的相似度计算，进而对图片进行排序，完成最终的图片查询任务。下面是一个简单的求两张图片余弦相似度的示例代码：
```python
import numpy as np

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_passvitb_image-reid-person_market'
input_location_1 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_reid_person.jpg'
input_location_2 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_reid_person.jpg'

image_reid_person = pipeline(Tasks.image_reid_person, model=model_id)
result_1 = image_reid_person(input_location_1)
result_2 = image_reid_person(input_location_2)

feat_1 = np.array(result_1[OutputKeys.IMG_EMBEDDING][0])
feat_2 = np.array(result_2[OutputKeys.IMG_EMBEDDING][0])
print(f'feat_1: {feat_1.shape}')
print(f'feat_2: {feat_2.shape}')
feat_norm_1 = feat_1 / np.linalg.norm(feat_1)
feat_norm_2 = feat_2 / np.linalg.norm(feat_2)
score = np.dot(feat_norm_1, feat_norm_2)
print(f'cosine score is: {score}')
```

### 模型局限性以及可能的偏差
本模型基于Market1501数据集进行finetune，在开放场景下的精度会下降。

## 训练数据介绍
本模型是基于[Market1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)数据集训练得到，该数据集为重识别领域常用数据集，包含了32668个矩形框标注，共1501个个体。该数据集具有以下两个特点：

- 数据集中的矩形框是使用Deformable Part Model (DPM)算法检测得到；
- 除了正确的矩形框，本数据集也保留了一些错误的矩形框作为干扰；

## 模型训练流程
本模型的训练过程主要分为**自监督预训练**和**有监督微调**两个阶段，目前暂不支持在线训练。


## 数据评估及结果
模型在Market1501的验证集上客观指标如下：
| Method | mAP | Rank-1 |
| ------------ | ------------ | ------------ |
| PASS ViT-B/16 | 0.933 | 0.969 |

### 相关论文以及引用信息
本模型主要参考论文如下（论文链接：[link](https://arxiv.org/abs/2203.03931)）：

```BibTeX
@article{zhu2022part,
  title={PASS: Part-Aware Self-Supervised Pre-Training for Person Re-Identification},
  author={Zhu, Kuan and Guo, Haiyun and Yan, Tianyi and Zhu, Yousong and Wang, Jinqiao and Tang, Ming},
  journal={arXiv preprint arXiv:2203.03931},
  year={2022}
}
```

