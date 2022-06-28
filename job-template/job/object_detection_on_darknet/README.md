# object_detection_on_darknet
描述：基于darknet框架实现的目标识别模板

镜像：ccr.ccs.tencentyun.com/cube-studio/object_detection_on_darknet:v1   

环境变量：  
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=2
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```

启动参数：  
```bash
{
    "args": {
        "--train_cfg": {
            "type": "text",
            "item_type": "str",
            "label": "模型参数配置、训练配置",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "[net]\n# Testing\n# batch=1\n# subdivisions=1\n# Training\nbatch=64\nsubdivisions=16\nwidth=608\nheight=608\nchannels=3\nmomentum=0.9\ndecay=0.0005\nangle=0\nsaturation = 1.5\nexposure = 1.5\nhue=.1\n\nlearning_rate=0.001\nburn_in=1000\nmax_batches = 501500\npolicy=steps\nsteps=400000,450000\nscales=.1,.1\n\n[convolutional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n# Downsample\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=32\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n# Downsample\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n# Downsample\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n# Downsample\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n# Downsample\n\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[shortcut]\nfrom=-3\nactivation=linear\n\n######################\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=1024\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=1024\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=1024\nactivation=leaky\n\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n\n\n[yolo]\nmask = 6,7,8\nanchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\nclasses=80\nnum=9\njitter=.3\nignore_thresh = .7\ntruth_thresh = 1\nrandom=1\n\n\n[route]\nlayers = -4\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[upsample]\nstride=2\n\n[route]\nlayers = -1, 61\n\n\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=512\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=512\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=512\nactivation=leaky\n\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n\n\n[yolo]\nmask = 3,4,5\nanchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\nclasses=80\nnum=9\njitter=.3\nignore_thresh = .7\ntruth_thresh = 1\nrandom=1\n\n\n\n[route]\nlayers = -4\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[upsample]\nstride=2\n\n[route]\nlayers = -1, 36\n\n\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=256\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=256\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=256\nactivation=leaky\n\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n\n\n[yolo]\nmask = 0,1,2\nanchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\nclasses=80\nnum=9\njitter=.3\nignore_thresh = .7\ntruth_thresh = 1\nrandom=1\n\n",
            "placeholder": "",
            "describe": "模型参数配置、训练配置",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--data_cfg": {
            "type": "text",
            "item_type": "str",
            "label": "训练数据配置",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "classes= 80\ntrain  = /app/coco_data_sample/trainvalno5k.txt\n#valid  = coco_testdev\nvalid = /app/coco_data_sample/5k.txt\nnames = /app/darknet/data/coco.names\nbackup = /app/backup #模型保存位置\neval=coco\n\n",
            "placeholder": "",
            "describe": "训练数据配置",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--weights": {
            "type": "text",
            "item_type": "str",
            "label": "权重文件",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/app/yolov3.weights",
            "placeholder": "",
            "describe": "权重文件",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```
