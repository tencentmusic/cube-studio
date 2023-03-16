
# RDE_VOS视频目标分割

## 任务
给定一个视频帧序列，和视频第一帧中想要分割的不同物体的掩码(mask)，模型会预测视频后续帧中对应物体的掩码(mask)

## 模型描述

本模型基于**Recurrent Dynamic Embedding for Video Object Segmentation**算法，是该算法的官方模型。

技术细节请见：

**Recurrent Dynamic Embedding for Video Object Segmentation** <br />
Mingxing Li, Li Hu, Zhiwei Xiong, Bang Zhang, Pan Pan, Dong Liu <br />
**CVPR 2022** <br />
**[[Paper](https://arxiv.org/abs/2205.03761)]**  <br />


<p float="left">
  &emsp;&emsp; <img src="description/rde.png" width="400" />
</p>

![Output1](description/dancing.gif)


## 如何使用

在ModelScope框架上，提供输入的视频帧序列、第一帧的mask标注，即可以通过简单的Pipeline调用来使用本模型，得到所有帧的mask预测结果。

### 代码示例

```python
import os
from PIL import Image
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import masks_visualization

task = 'video-object-segmentation'
model_id = 'damo/cv_rdevos_video-object-segmentation'
input_location = 'data/test/videos/video_object_segmentation_test'
images_dir = os.path.join(input_location, 'JPEGImages')
mask_file = os.path.join(input_location, 'Annotations', '00000.png')

input_images = []
for image_file in sorted(os.listdir(images_dir)):
    img = Image.open(os.path.join(images_dir, image_file)).convert('RGB')
    input_images.append(img)
mask = Image.open(mask_file).convert('P')
input = {'images': input_images, 'mask': mask}

segmentor = pipeline(
    Tasks.video_object_segmentation, model=model_id)
result = segmentor(input)
out_masks = result[OutputKeys.MASKS]

vis_masks = masks_visualization(out_masks, mask.getpalette())

os.makedirs('test_result', exist_ok=True)
for f, vis_mask in enumerate(vis_masks):
    vis_mask.save(os.path.join('test_result', '{:05d}.png'.format(f)))

print('test_video_object_segmentation DONE')
```



## 模型精度
在YoutubeVOS 2019上的结果为
| Dataset | Split | J&F | J_Seen | F_Seen | J_Unseen | F_Unseen |
| --- | --- | :--:|:--:|:---:|:---:|:---:|
| YouTube 2019| validation | 83.3| 81.9|86.3|78.0|86.9|



## Bibtex

```
@inproceedings{li2022recurrent,
  title={Recurrent Dynamic Embedding for Video Object Segmentation},
  author={Li, Mingxing and Hu, Li and Xiong, Zhiwei and Zhang, Bang and Pan, Pan and Liu, Dong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1332--1341},
  year={2022}
}
```