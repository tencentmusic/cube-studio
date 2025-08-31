import argparse
import json
import logging
import math
import os,sys,re,argparse
import random
import shutil
import time
import pysnooper
from ultralytics import YOLO
import math
import datetime
import pandas as pd
import yaml
import faulthandler, torch, cv2
faulthandler.enable()
torch.backends.mkldnn.enabled = False
cv2.ocl.setUseOpenCL(False)


base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)
sys.path.append(os.path.realpath(__file__))

yolo_classes = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

# python train.py --train /mnt/admin/coco_data_sample/train.txt --val /mnt/admin/coco_data_sample/valid.txt --batch_size 1 --epoch 1 --save_model_path /mnt/admin/coco_data_sample/yolov8_best.pt --weights yolov8n.pt
# @pysnooper.snoop()
def main():
    arg_parser = argparse.ArgumentParser("obj launcher")
#    # XGBClassifier XGBRegressor
    arg_parser.add_argument('--train', type=str, help="coco训练数据地址", default='')
    arg_parser.add_argument('--val', type=str, help="训练数据地址", default='')
    arg_parser.add_argument('--classes', type=str, help="分类类型，逗号分割", default='')
    arg_parser.add_argument('--weights', type=str, help="预训练权重文件地址", default='/yolov8/yolov8n.pt')
    arg_parser.add_argument('--epochs', type=int, default=300)
    arg_parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    arg_parser.add_argument('--worker', type=int, default=8, help='数据加载并发数')
    arg_parser.add_argument('--img_size',  type=str, default="640", help='[train, test] image sizes')
    arg_parser.add_argument('--save_model_path', type=str, help="训练模型保存地址", default='')

    args = arg_parser.parse_args()
    # # 删除历史模型，训练成功会自动删除，
    # if os.path.exists(args.save_model_path) and os.path.isfile(args.save_model_path):
    #     os.remove(args.save_model_path)
    # 创建目录
    os.makedirs(os.path.dirname(args.save_model_path),exist_ok=True)

    WORKFLOW_NODE_NAME = os.getenv('WORKFLOW_NODE_NAME', "")
    first = not WORKFLOW_NODE_NAME.endswith(")") or WORKFLOW_NODE_NAME.endswith("(0)")

    # 识别设备
    device = 'cpu'
    resource_gpu = os.getenv('KFJ_TASK_RESOURCE_GPU','')
    resource_gpu = resource_gpu.split('(')[0]
    resource_gpu = resource_gpu.split(',')[-1]
    resource_gpu = float(resource_gpu) if resource_gpu else 0
    if resource_gpu>0:
        device='0'
        # resource_gpu = math.ceil(resource_gpu)
        # resource_gpu = list(range(resource_gpu))
        # resource_gpu = [str(x) for x in resource_gpu]
        # device = ','.join(resource_gpu)

    if ',' in args.img_size or '，' in args.img_size:
        img_size = re.split(',|，', args.img_size)
        img_size = [int(x.strip()) for x in img_size if x.strip()]
    else:
        img_size = int(args.img_size.strip())

    # 生成模型训练yaml文件
    logging.info("{} args: {}".format(__file__, args))
    data_config = open('/yolov8/yolov8.yaml').read()
    classes = args.classes.split(',')
    classes = [x.strip() for x in classes if x.strip()]
    if not classes:
        classes = yolo_classes
    classes = [f"  {index}: {class1}" for index, class1 in enumerate(classes)]
    classes = '\n'.join(classes)
    if not args.val:
        data_config.replace('val: VAL_DATASET', '')
    data_config = data_config.replace('TRAIN_DATATSE', args.train).replace('VAL_DATASET', args.val).replace('CLASSES',
                                                                                                            str(classes))
    with open('/yolov8/data.yaml', 'w') as f_data_cfg:
        f_data_cfg.write(data_config)

    # 规定训练文件存放目录
    train_file_path = os.path.join(os.path.dirname(args.save_model_path), 'tensorboard')
    if not first:
        # 继续训练上次的结果
        last_model_path = None
        # 得出last.pt文件地址
        for root, dirs, files in os.walk(train_file_path):
            for file in files:
                if file == 'last.pt':
                    last_model_path = os.path.join(root, file)
                    print(f"[LOG] 找到上次未完成训练模型地址 {last_model_path}")
                    break
        if last_model_path:
            print("[LOG] 开始继续上次中断训练")
            model = YOLO(model=last_model_path)
            model.train(resume=True,
                        data='/yolov8/data.yaml',
                        epochs=int(args.epochs),
                        imgsz=img_size,
                        batch=int(args.batch_size),
                        device=device,
                        project=train_file_path,
                        patience=int(args.epochs))
    else:
        # 新的训练,先清空旧文件夹
        shutil.rmtree(train_file_path, ignore_errors=True)
        # Load a model
        model = YOLO(model=args.weights)  # load a pretrained model (recommended for training)

        # shutil.rmtree(train_file_path)
        # os.system(f"tensorboard --logdir {train_file_path} &")

        model.train(data='/yolov8/data.yaml',
                    epochs=int(args.epochs),
                    imgsz=img_size,
                    batch=int(args.batch_size),
                    device=device,
                    project=train_file_path,
                    patience=int(args.epochs))

    for root, dirs, files in os.walk(train_file_path):
        for file in files:
            if file=='best.pt':
                model_path = os.path.join(root, file)
                if os.path.exists(args.save_model_path):
                    os.remove(args.save_model_path)
                shutil.copy(model_path,args.save_model_path)

                # 编写结果可视化
                metric = {
                    "metric_type":"image",
                    "describe":"训练评估指标",
                    "image":os.path.join(train_file_path,'train','results.png')
                  }
                json.dump(metric,open('/metric.json',mode='w'))

                exit(0)
    print('未发现最优模型')
    exit(1)

if __name__ == "__main__":
    main()
