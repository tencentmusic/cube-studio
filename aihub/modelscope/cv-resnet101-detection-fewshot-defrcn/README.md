
## 模型描述
少样本目标检测模型DeFRCN，提出了一种简单而有效的基于Decoupled Faster R-CNN，引入新的GDL和PCB，显著地缓解了传统Faster R-CNN在数据匮乏场景下的潜在问题。

![模型信息](description/header.png)

模型结构

![模型结构](description/arch.png)

## 环境依赖
推荐基于ModelScope官方镜像使用，[获取地址](https://www.modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)
在此基础上需要安装detectron2-0.3/gpu版本，cpu版本待提供。
```
pip install detectron2==0.3 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

## 使用方法
如果模型配置文件中的test.pcb_enable为true，将会启用PCB模块，需要使用原始图像提取特征，以获得更准确的分类结果。
因此会使用模型配置文件中的datasets相关配置，datasets.root如果为null，将下载默认数据集进行计算。
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

defrcn_detection = pipeline(Tasks.image_fewshot_detection, 'damo/cv_resnet101_detection_fewshot-defrcn')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_voc2007_000001.jpg'
result = defrcn_detection(img_path)

print(result)
```

## 数据集
支持 pascal_voc 和 coco2014 两个fewshot数据集，也可以根据需要修改源码，添加自己的数据集。
数据详情见:
    [VOC_fewshot](https://www.modelscope.cn/datasets/shimin2023/VOC_fewshot)
    [coco2014_fewshot](https://www.modelscope.cn/datasets/shimin2023/coco2014_fewshot)


## 模型finetune
先训练一个base pretrain model，然后在这个模型基础上再训练few shot模型。
下面的用例是通过使用托管在modelscope DatasetHub上的数据集VOC_fewshot进行训练：

第一步：准备训练数据集和预训练模型
```python
import os
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers
from modelscope.utils.constant import DownloadMode
from modelscope.hub.snapshot_download import snapshot_download


# -----------------------下载pretrain model---------------------
model_id = 'damo/cv_resnet101_detection_fewshot-defrcn'
model_dir = snapshot_download(model_id, cache_dir='./')

# ------------------------VOC fewshot训练数据下载---------------------------
cache_data_dir = './damo/datasets/'
data_voc = MsDataset.load(
            dataset_name='VOC_fewshot',
            namespace='shimin2023',
            split='train',
            cache_dir=cache_data_dir,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
data_dir = os.path.join(data_voc.config_kwargs['split_config']['train'], 'data')

```
第二步：训练base pretrain model
```python
output_dir = "./damo/defrcn_output/"
DATA_TYPE = 'pascal_voc' # 目前仅支持pascal_voc和coco
split = 1 # [1,2,3]，数据集进行了三种划分，均可以进行验证

# ------------------------base pretrain-------------------------
print("start training base model")
base_work_dir = os.path.join(output_dir, 'defrcn_det_r101_base{}'.format(split))
def base_cfg_modify_fn(cfg):
    cfg.train.work_dir = base_work_dir
    cfg.model.weights = os.path.join(model_dir, 'ImageNetPretrained/MSRA/R-101.pkl')
    cfg.model.roi_heads.num_classes = 15
    cfg.model.roi_heads.backward_scale = 0.75
    cfg.model.roi_heads.freeze_feat = False
    cfg.model.roi_heads.cls_dropout = False
    cfg.datasets.root = data_dir
    cfg.datasets.type = DATA_TYPE
    cfg.datasets.train = ["voc_2007_trainval_base{}".format(split), 'voc_2012_trainval_base{}'.format(split)]
    cfg.datasets.test = ['voc_2007_test_base{}'.format(split)]
    cfg.test.pcb_enable = False
    cfg.train.dataloader.ims_per_batch = 16
    cfg.train.optimizer.lr = 0.02
    cfg.train.lr_scheduler.steps = [10000,13300]
    cfg.train.max_iter = 15000
    cfg.train.lr_scheduler.warmup_iters = 100
    return cfg

base_kwargs = dict(model=model_id, cfg_modify_fn=base_cfg_modify_fn)
base_trainer = build_trainer(name=Trainers.image_fewshot_detection, default_args=base_kwargs)
base_trainer.train()

base_trainer.model_surgery(os.path.join(base_work_dir, 'model_final.pth'), base_work_dir, data_type=DATA_TYPE, method='remove') # 构建fsod pretrain model
fsod_base_weight = os.path.join(base_work_dir, 'model_reset_remove.pth')

base_trainer.model_surgery(os.path.join(base_work_dir, 'model_final.pth'), base_work_dir, data_type=DATA_TYPE, method='randinit') # 构建gfsod pretrain model
gfsod_base_weight = os.path.join(base_work_dir, 'model_reset_surgery.pth')
```
第三步：训练fsod模型/gfsod模型
```python
# --------------------------fsod fine-tuning------------------------
print("start training fsod model")

shot = 10 # [1, 2, 3, 5, 10]
seed = 0

fsod_work_dir = os.path.join(output_dir, 'defrcn_fsod_r101_novel{}/{}shot_seed{}'.format(split, shot, seed))
def fsod_cfg_modify_fn(cfg):
    cfg.train.work_dir = fsod_work_dir
    cfg.datasets.root = data_dir
    cfg.datasets.type = DATA_TYPE
    cfg.model.weights = fsod_base_weight
    cfg.model.roi_heads.num_classes = 5
    cfg.datasets.train = ["voc_2007_trainval_novel{}_{}shot_seed{}".format(split, shot, seed)]
    cfg.datasets.test = ['voc_2007_test_novel{}'.format(split)]
    cfg.test.pcb_modelpath = os.path.join(model_dir, 'ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth')
    return cfg

fsod_kwargs = dict(model=model_id, cfg_modify_fn=fsod_cfg_modify_fn)
fsod_trainer = build_trainer(name=Trainers.image_fewshot_detection, default_args=fsod_kwargs)
fsod_trainer.train()
metrics = fsod_trainer.evaluate("{}/model_final.pth".format(fsod_work_dir))
print(metrics)
```
```python
# --------------------------gfsod fine-tuning-------------------------
print("start training gfsod model")

seed = 0 # [0,1,2,3,4,5,6,7,8,9]
shot = 10 # [1, 2, 3, 5, 10]
gfsod_work_dir = os.path.join(output_dir, 'defrcn_gfsod_r101_novel{}/{}shot_seed{}'.format(split, shot, seed))
def gfsod_cfg_modify_fn(cfg):
    cfg.train.work_dir = gfsod_work_dir
    cfg.datasets.root = data_dir
    cfg.datasets.type = DATA_TYPE
    cfg.model.weights = gfsod_base_weight
    cfg.model.roi_heads.num_classes = 20
    cfg.datasets.train = ["voc_2007_trainval_all{}_{}shot_seed{}".format(split, shot, seed)]
    cfg.datasets.test = ["voc_2007_test_all{}".format(split)]
    cfg.test.pcb_modelpath = os.path.join(model_dir, 'ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth')
    return cfg

gfsod_kwargs = dict(model=model_id, cfg_modify_fn=gfsod_cfg_modify_fn)
gfsod_trainer = build_trainer(name=Trainers.image_fewshot_detection, default_args=gfsod_kwargs)
gfsod_trainer.train()
metrics = gfsod_trainer.evaluate("{}/model_final.pth".format(gfsod_work_dir))
print("{}".format(metrics))
```

## 来源说明
模型方法基于[DeFRCN](https://github.com/er-muyue/DeFRCN)，请遵守相关许可。

## 引用
```
 @inproceedings{qiao2021defrcn,
  title={DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection},
  author={Qiao, Limeng and Zhao, Yuxuan and Li, Zhiyuan and Qiu, Xi and Wu, Jianan and Zhang, Chi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8681--8690},
  year={2021}
}
```