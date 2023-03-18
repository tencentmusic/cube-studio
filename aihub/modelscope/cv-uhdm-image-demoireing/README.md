
# uhdm-image-demoireing模型介绍
给定一张输入带有摩尔纹的图像，输出去除摩尔纹图像；

拍摄数字屏幕上显示的内容时，相机的颜色滤波器阵列(CFA)和屏幕LCD亚像素之间存在频率堆叠效应，呈现的图像混合了彩色的条纹，此类图像称为摩尔纹图像。

<img src=resources/test_moire_0.jpg width=25% /><img src=resources/test_moire_0_demoire.jpg width=25% />

## 模型结构
模型主干是编码-解码网络，同时在不同语义特征层堆叠语义特征对齐感知模块SAM，以提升模型处理大分辨率摩尔纹尺度变化的能力。

<img src=resources/esdnet.jpg width=25% /><img src=resources/esdnet_cmp.jpg width=25% />


## 期望模型使用方式与适用范围
本模型主要是针对优化摩尔纹图像的质量，具有一定的领域适用性。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
image_demoire = pipeline(Tasks.image_demoireing, model='damo/cv_uhdm_image-demoireing')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_moire.jpg'
result = image_demoire(img_path)
from PIL import Image
Image.fromarray(result[OutputKeys.OUTPUT_IMG]).save('./result.jpg')
```
### 模型局限性以及可能的偏差
- 算法在训练同分布数据范围内可以取得良好的修复效果，还有一定的优化提升空间。
## 训练数据介绍
- UHDM
  详情见：https://xinyu-andy.github.io/uhdm-page/
### 预处理
- 给定一张输入图像，图像分辨率可32倍整除归一化。
## 数据评估及结果
| DataSet  |     PSNR         |  SSIM           |      LPIPS       |
|:--------:|:----------------:|:---------------:|:----------------:|
| UHDM     | 22.119/22.422    |  0.7956/0.7985  |   0.2551/0.2454  |
| FHDMi    | 24.500/24.882    |  0.8351/0.8440  |   0.1354/0.1301  |   
| TIP2018  | 29.81 /30.11     |  0.916 /0.920   |   --- /---       |  
| LCDMoire | 44.83 /45.34     |  0.9963/0.9966  |   --- /---       |
## 引用
```BibTeX
@inproceedings{Yu2022TowardsEA,
  title={Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing},
  author={Xin Yu and Peng Dai and Wenbo Li and Lan Ma and Jiajun Shen and Jia Li and Xiaojuan Qi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
@inproceedings{dai2022video,
  title={Video Demoireing with Relation-Based Temporal Consistency},
  author={Dai, Peng and Yu, Xin and Ma, Lan and Zhang, Baoheng and Li, Jia and Li, Wenbo and Shen, Jiajun and Qi, Xiaojuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
