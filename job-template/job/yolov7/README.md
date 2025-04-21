# yolov7
描述：基于yolov7框架实现的目标识别模板

镜像：ccr.ccs.tencentyun.com/cube-studio/yolov7:2024.01

启动参数：  
```bash
{
    "训练参数": {
        "--train": {
            "type": "str",
            "item_type": "str",
            "label": "训练数据集",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/{{creator}}/coco_data_sample/train.txt",
            "placeholder": "",
            "describe": "训练数据集，txt配置地址",
            "editable": 1
        },
        "--val": {
            "type": "str",
            "item_type": "str",
            "label": "验证数据集",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/{{creator}}/coco_data_sample/valid.txt",
            "placeholder": "",
            "describe": "验证数据集，txt配置地址",
            "editable": 1
        },
        "--classes": {
            "type": "text",
            "item_type": "str",
            "label": "目标分类",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,trafficlight,firehydrant,stopsign,parkingmeter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sportsball,kite,baseballbat,baseballglove,skateboard,surfboard,tennisracket,bottle,wineglass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hotdog,pizza,donut,cake,chair,couch,pottedplant,bed,diningtable,toilet,tv,laptop,mouse,remote,keyboard,cellphone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddybear,hairdrier,toothbrush",
            "placeholder": "",
            "describe": "目标分类,逗号分割",
            "editable": 1
        },
        "--batch_size": {
            "type": "str",
            "item_type": "str",
            "label": "batch-size",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "1",
            "placeholder": "",
            "describe": "batch-size",
            "editable": 1
        },
        "--epoch": {
            "type": "str",
            "item_type": "str",
            "label": "epoch",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "1",
            "placeholder": "",
            "describe": "epoch",
            "editable": 1
        },
        "--weights": {
            "type": "str",
            "item_type": "str",
            "label": "权重文件",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/yolov7/weights/yolov7_training.pt",
            "placeholder": "",
            "describe": "权重文件",
            "editable": 1
        },
        "--save_model_path": {
            "type": "str",
            "item_type": "str",
            "label": "模型保存地址",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/{{creator}}/coco_data_sample/yolov7_best.pt",
            "placeholder": "",
            "describe": "模型保存地址",
            "editable": 1
        }
    }
}
```
