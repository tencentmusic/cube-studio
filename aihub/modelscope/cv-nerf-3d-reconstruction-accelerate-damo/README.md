
NeRF(Neural Radiance Fields)是一种利用多视角图像进行三维重建的技术，其通过隐式表征的方式来对静态三维物体或场景进行学习和建模。
NeRF快速三维重建模型，能够快速（~10min）对物体进行三维重建和新视角合成。

## 效果展示
以下是在[nerf-synthesis](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) 数据集上使用NeRF快速三维重建模型重建渲染的结果。


<div align=center>
<img src="description/lego.gif">
</div>

<div align=center>
<img src="description/chair.gif">
</div>

## 模型描述

该模型通过对NeRF端到端重建pipeline中的编码，存储，渲染进行优化，极大提升了NeRF网络重建的训练速度，将单个物体的重建开销由10hour+提升
到~10min内。

其中使用的相关加速技术和实现包括instant-ngp的Multiresolution Hash Encoding技术，nerfacc框架的高效体素渲染技术和tinycudan的pytorch扩展实现。

## 使用方式和范围

使用范围:
当前支持两种数据类型，分别是[nerf-synthesis](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) 和用户自定义数据。
- nerf-synthesis数据集人工合成的多视角数据集，包含8个场景
  下载及更多信息请参考[NeRF三维重建数据集](https://www.modelscope.cn/datasets/damo/nerf_recon_dataset/summary)
- 用户自定义数据分为图像集（将图像集合保存到images文件夹）和单个视频输入（支持主流视频格式）

目标场景:
- 需要对单一静态物体进行三维重建的场景
- 对nerf重建加速有需求的场景

运行环境：
- 模型只支持GPU上运行，已在p100, v100, RTX3090卡上测试通过，具体效率与gpu性能相关，可能存在一些差异。
- GPU显存要求大于等于16G.


### 如何使用

#### 数据采集

想要对自摄物体进行重建时，用户仅需使用手机等便捷手持设备进行数据采集。采集过程需要对重建的物体进行多视角的拍摄，
保证物体保持不动，相机围绕物体进行多视角的移动拍摄，尽量覆盖多样视角，并保证视频长度大于等于4s(提取的图像帧>=60）。


#### 新视角渲染
在ModelScope框架上，训练好一个NeRF模型后即可以通过简单的Pipeline调用来使用。(仅支持GPU运行)

#### 推理代码范例

以lego场景训练的模型为例，其属于nerf-synthesis数据集，需要指定data_type='blender', 
提供数据目录，使用默认的预训练模型即可渲染出新视角的结果，结果保存在用户自定义的render_dir目录下。
如下代码可以直接运行。

```python
import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.msdatasets import MsDataset


data_dir = MsDataset.load(
    'nerf_recon_dataset', namespace='damo',
    split='train').config_kwargs['split_config']['train']
nerf_synthetic_dataset = os.path.join(data_dir, 'nerf_synthetic')
blender_scene = 'lego'
data_dir = os.path.join(nerf_synthetic_dataset, blender_scene)
render_dir = 'exp'

### when use nerf-synthesis dataset, data_type should specify as 'blender'
nerf_recon_acc = pipeline(
    Tasks.nerf_recon_acc,
    model='damo/cv_nerf-3d-reconstruction-accelerate_damo',
    data_type='blender',
    )

nerf_recon_acc(
    dict(data_dir=data_dir, render_dir=render_dir))
### render results will be saved in render_dir
```

如使用自定义数据集，则需要指定data_type='colmap', 
推理方式及更多自定义参数如下示例。如下代码无法直接运行，具体参数需要用户自行指定。

```python
import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

data_dir = 'PATH/TO/Data'
render_dir = 'exp'

### when use nerf-synthesis dataset, data_type should specify as 'blender'
nerf_recon_acc = pipeline(
    Tasks.nerf_recon_acc,
    model='damo/cv_nerf-3d-reconstruction-accelerate_damo',
    data_type='colmap',
    ckpt_path='/PATH/TO/CHECKPOINT/model.ckpt',
    save_mesh=False,
    n_test_traj_steps=120,
    test_ray_chunk=1024
)

nerf_recon_acc(
    dict(data_dir=data_dir, render_dir=render_dir))
### render results will be saved in render_dir
```

#### 训练代码范例


nerf-systhesis数据集的训练代码示例

```
import os
from modelscope.msdatasets import MsDataset
from modelscope.trainers.cv import NeRFReconAccTrainer
from modelscope.utils.test_utils import test_level

model_id = 'damo/cv_nerf-3d-reconstruction-accelerate_damo'
data_dir = MsDataset.load(
        'nerf_recon_dataset', namespace='damo',
        split='train').config_kwargs['split_config']['train']

trainer = NeRFReconAccTrainer(model=model_id,
                              data_type='blender',
                              work_dir='exp_nerf_synthetic',
                              render_images=False)

nerf_synthetic_dataset = os.path.join(data_dir, 'nerf_synthetic')
blender_scene = 'lego'  ## can choose any of the 8 scenes
nerf_synthetic_dataset = os.path.join(nerf_synthetic_dataset,
                                      blender_scene)
trainer.train(data_dir=nerf_synthetic_dataset)
```

用户提供图像数据的训练代码示例

```
import os
from modelscope.msdatasets import MsDataset
from modelscope.trainers.cv import NeRFReconAccTrainer
from modelscope.utils.test_utils import test_level

model_id = 'damo/cv_nerf-3d-reconstruction-accelerate_damo'
data_dir = MsDataset.load(
        'nerf_recon_dataset', namespace='damo',
        split='train').config_kwargs['split_config']['train']

trainer = NeRFReconAccTrainer(model=model_id,
                              data_type='colmap',
                              work_dir='exp_nerf_image',
                              render_images=False)

custom_dir = os.path.join(data_dir, "custom/hotdog")
trainer.train(data_dir=custom_dir)
```

用户提供视频数据的训练代码示例

```
import os
from modelscope.trainers.cv import NeRFReconAccTrainer
from modelscope.utils.test_utils import test_level

model_id = 'damo/cv_nerf-3d-reconstruction-accelerate_damo'
trainer = NeRFReconAccTrainer(model=model_id,
                              data_type='colmap',
                              work_dir='exp_nerf_video',
                              render_images=False)

video_input_path = "**.mp4"
trainer.train(video_input_path=video_input_path)
```

### 数据预处理

- 如输入为视频，则会将视频拆分为图像帧。
- 对图像集合进行相机视角估计。
- 对图像中的物体进行分割。

### 模型局限性以及可能的偏差

以下是针对用户自定义数据场景的说明

- 模型预处理依赖colmap对视频帧进行相机视角估计，该估计过程对不同场景存在一定误差，可能影响重建渲染结果。

- 模型数据预处理依赖图像分割模型，如拍摄物体边缘结果复杂或者背景存在干扰导致无法正确分割出物体，将影响到重建渲染效果。

- 测试渲染视角为随机采样，如果用户采集数据时覆盖视角较少，可能导致测试视角与拍摄视角存在较大差异，将会影响测试视角的渲染效果。



## 数据评估及结果

以下数据评估为该模型在[nerf-synthesis](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) 
数据集上的评测效果，测试机器为单卡RTX3090.

| Name | ship | mic | materials | lego | hotdog | ficus | drums | chairs |
| ------------ | ------------ | ------------ |------------ | ------------ | ------------ | ------------ |------------ | ------------ |
| PSNR | 30.03 | 35.48 | 29.30 | 35.26 | 37.2 | 33.83 | 25.83 | 35.10 |
| Recon Time(s) | 437 | 177 | 248 | 214 | 278 | 187 | 213 | 181 |

## 相关工作
```BibTeX
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}

@misc{tiny-cuda-nn,
    Author = {Thomas M\"uller},
    Year = {2021},
    Note = {https://github.com/nvlabs/tiny-cuda-nn},
    Title = {Tiny {CUDA} Neural Network Framework}
}

@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}

@article{li2022nerfacc,
  title={NerfAcc: A General NeRF Accleration Toolbox.},
  author={Li, Ruilong and Tancik, Matthew and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2210.04847},
  year={2022}
}
```