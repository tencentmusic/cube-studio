
# 基于层次化表征的高精度人脸重建模型

### [论文](https://arxiv.org/abs/2302.14434)

人脸重建模型以单张人像图作为输入，利用层次化表征实现快速人脸几何、纹理恢复，输出高精度3D人脸重建mesh，相关论文已被CVPR2023接收。


![Output1](results/result_1.gif)

![Output2](https://modelscope.cn/api/v1/models/damo/cv_resnet50_face-reconstruction/repo?Revision=master&FilePath=results/result_2.jpg&View=true)

## 模型描述

该人脸重建模型以deep3d为基础，进行了以下改进，以实现更高精度的人脸重建：
- 融合BFM与FLAME，构建新的head 3DMM模型，结合两者的头部完整性、脸部高精度等优势。
- 引入deformation map等表征，实现层次化建模，提升人脸的重建细节。
- 引入contour loss，提升脸型重建精度。

模型相关方法 [HRN](https://arxiv.org/abs/2302.14434) 已被CVPR2023接收，网络结构如下图。与HRN略有差异的是，目前该模型采用inference+fitting的方式进行重建，针对输入图进行finetune以实现更为精确的重建效果。

![内容图像](https://modelscope.cn/api/v1/models/damo/cv_resnet50_face-reconstruction/repo?Revision=master&FilePath=assets/HRN_framework.jpg&View=true)

## 期望模型使用方式以及适用范围

使用方式：
- 可直接使用模型进行推理。具体的，对于单张图像的重建包含回归+拟合两个部分组成，回归部分直接由模型推理预测3DMM系数得到coarse mesh，而后采用拟合的方式进一步预测deformation map，对coarse mesh进行变形得到更精准的mesh。

使用范围:
- 适用于包含人脸的人像照片，其中人脸分辨率大于100x100，图像整体分辨率小于5000x5000。

目标场景:
- 影视、娱乐、医美等。

### 如何使用

本模型基于pytorch进行训练和推理，在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来使用人脸重建模型。

#### 环境安装
安装好基础modelscope环境后，安装nvdiffrast
```
# 安装nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .

# 安装nvdiffrast所需依赖（opengl等）
apt-get install freeglut3-dev
apt-get install binutils-gold g++ cmake libglew-dev mesa-common-dev build-essential libglew1.5-dev libglm-dev
apt-get install mesa-utils
apt-get install libegl1-mesa-dev 
apt-get install libgles2-mesa-dev
apt-get install libnvidia-gl-525
```

#### 代码范例
当前模型依赖gpu进行3D渲染，请在gpu环境进行试用、测试
```python
from modelscope.models.cv.face_reconstruction.utils import write_obj
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

face_reconstruction = pipeline(Tasks.face_reconstruction,model='damo/cv_resnet50_face-reconstruction')
result = face_reconstruction('data/test/images/face_reconstruction.jpg')
mesh = result[OutputKeys.OUTPUT]
write_obj('result_face_reconstruction.obj', mesh)
```

### 模型局限性以及可能的偏差
- 在人脸分辨率大于100×100的图像上可取得期望效果，分辨率过小时会导致重建结果精度较差、纹理不清晰。
- 对于脸部存在遮挡的情况，重建效果可能不理想。

## 训练数据介绍
- 本模型通过自监督的方式进行训练，训练数据仅需2D人脸图像，可使用人脸公开数据集如CeleA、300W-LP等。

### 预处理
- 人脸区域裁剪、resize到224x224分辨率作为网络输入。

### 后处理
- 将顶点坐标、三角面片、贴图等数据转化为obj等模型文件。

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{Lei2023AHR,
  title={A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images},
  author={Biwen Lei and Jianqiang Ren and Mengyang Feng and Miaomiao Cui and Xuansong Xie},
  year={2023}
}
```
