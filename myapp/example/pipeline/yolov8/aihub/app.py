
import io,sys,os,base64,pysnooper
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy
import os,logging
import datetime
import pysnooper
import requests
import logging
import time
from PIL import Image
from ultralytics import YOLO
from flask import jsonify
from cubestudio.aihub.web.labelstudio import LabelStudio_ML_Backend

device='cpu'

# 加载模型
# 这里是添加的gpu识别
resource_gpu = os.getenv('RESOURCE_GPU', '')
resource_gpu = resource_gpu.split('(')[0]
resource_gpu = int(resource_gpu) if resource_gpu else 0
if resource_gpu:
    device = 'cuda:0'

class YOLOV8_Model(Model,LabelStudio_ML_Backend):
    # 模型基础信息定义
    name='yolov8'   # 该名称与目录名必须一样，小写
    label='yolov8目标识别'
    describe="yolov8目标识别，图像分割等"
    field="机器视觉"  # [机器视觉，听觉，自然语言，多模态，大模型]
    scenes="图像识别"
    status='online'
    images = 'ccr.ccs.tencentyun.com/cube-studio/yolov8:20250801'
    version='v20241001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = [
        Field(Field_type.text, name='--train', label='训练数据集', describe='训练数据集，txt配置地址',
              default='/mnt/{{creator}}/coco_data_sample/train.txt',validators=Validator()),
        Field(Field_type.text, name='--val', label='验证数据集', describe='验证数据集，txt配置地址',
              default='/mnt/{{creator}}/coco_data_sample/valid.txt', validators=Validator()),
        Field(Field_type.text, name='--classes', label='目标分类', describe='目标分类,逗号分割',
              default='person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,trafficlight,firehydrant,stopsign,parkingmeter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sportsball,kite,baseballbat,baseballglove,skateboard,surfboard,tennisracket,bottle,wineglass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hotdog,pizza,donut,cake,chair,couch,pottedplant,bed,diningtable,toilet,tv,laptop,mouse,remote,keyboard,cellphone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddybear,hairdrier,toothbrush', validators=Validator()),
        Field(Field_type.text, name='--batch_size', label='batch-size', describe='batch-size',
              default='1', validators=Validator()),
        Field(Field_type.text, name='--epochs', label='epochs', describe='epochs',
              default='1', validators=Validator())
    ]
    inference_outputs = ['image']
    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='待推理图片',default='', describe='待推理图片',validators=Validator())
    ]

    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例一描述",
            "input": {
                "image": "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000000597.jpg"
            }
        }
    ]

    # 准备默认数据
    def set_dataset(self,save_dataset_dir=None):
        # 此示例数据已经处理成规范结构
        dataset_dir = 'dataset'
        if save_dataset_dir:
            dataset_dir = save_dataset_dir
        os.makedirs(dataset_dir,exist_ok=True)
        if not os.path.exists("coco.zip"):
            os.system("wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/coco.zip")
        os.system(f'unzip -n -o -d {dataset_dir} coco.zip')
        os.system('rm coco.zip')


    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self,save_model_dir,train,val,classes,batch_size,epochs, **kwargs):
        print(save_model_dir,train,val,classes,batch_size,epochs,kwargs)
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下
        model_path = os.path.join(save_model_dir,'best.pt')
        os.system(f"python train.py --train {train} --val {val} --classes {classes} --batch_size {batch_size} --epoch {epochs} --weights /yolov8/yolov8n.pt --save_model_path {model_path}")
        return save_model_dir


    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    from cubestudio.aihub.model import check_has_load
    @check_has_load
    def load_model(self,save_model_dir=None,**kwargs):
        if save_model_dir:
            # 从train函数的目录下读取模型
            model_path = os.path.join(save_model_dir,'best.pt')
            self.model = YOLO(model_path)
            return
        else:
            model_path = os.getenv('MODELPATH',kwargs.get('model_path',''))
            if not model_path:
                current_directory = os.getcwd()
                model_extensions = ['.pt', '.pth']

                # 获取当前目录下的所有文件
                files_in_directory = os.listdir(current_directory)

                # 筛选出可能的模型文件
                model_files = [file for file in files_in_directory if os.path.splitext(file)[1] in model_extensions]
                if model_files:
                    self.model = YOLO(model_files[0])
                    return
            elif os.path.exists(model_path):
                self.model = YOLO(model_path)
                return

        print(f'未发现模型地址环境变量MODELPATH，或者模型文件{model_path}不存在')
        exit(1)


    # 自动化标注
    def labelstudio_predict(self,tasks,project,label_config,force_reload=False,try_fetch=True,params={},model_version=None,**kwargs):
        back = []
        for task in tasks:
            image_path = task['data']['image']  # 获取图片路径
            image_path = self.labelstudio_download_image(image_path)
            results = self.model(image_path, device=device)
            result = None
            if results:
                result = results[0]
            else:
                return {}

            names = result.names
            boxes = result.boxes
            orig_height, orig_width = boxes.orig_shape[0], boxes.orig_shape[1]
            xywhns = boxes.xywhn.tolist()
            xywhns = [[round(float(box[0]), 6), round(float(box[1]), 6), round(float(box[2]), 6), round(float(box[3]), 6)] for box in xywhns]
            # 把类型序号换成label名称
            cls = boxes.cls.tolist()
            cls = [names[index] for index in cls]
            conf = boxes.conf.tolist()
            predictions = {
                "names": list(names.values()),
                "labels": cls,
                "scores": conf,
                "xywhns": xywhns,
                "orig_shape": [orig_width, orig_height]
            }

            result = []
            labels = predictions['labels']
            scores = predictions['scores']
            xywhns = predictions['xywhns']
            height, width = predictions['orig_shape'][0], predictions['orig_shape'][1]

            import xml.etree.ElementTree as ET
            root = ET.fromstring(label_config)
            choices = []
            # 遍历所有的item节点
            for item in root.findall('.//Label'):  # 这里'.//item'是XPath，用于查找所有的item节点
                # 获取指定属性
                attribute_value = item.get('value')  # 这里'attribute'是你想要获取的属性名
                print(attribute_value)
                if attribute_value:
                    choices.append(attribute_value)
            choices = list(set(choices))
            choices_lower = [x.lower() for x in choices]

            for index, label in enumerate(labels):
                if label.lower() not in choices_lower:
                    continue
                label = choices[choices_lower.index(label.lower())]

                xywhn = xywhns[index]
                # 换成coco数据集格式 start_x,start_y,width,height
                xywhn = [int((xywhn[0] - xywhn[2] / 2) * 100), int((xywhn[1] - xywhn[3] / 2) * 100), int(xywhn[2] * 100), int(xywhn[3] * 100)]
                score = scores[index]
                result.append({
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": xywhn[0],
                        "y": xywhn[1],
                        "width": xywhn[2],
                        "height": xywhn[3],
                        "rotation": 0,
                        "rectanglelabels": [label]
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": "prediction",
                    "score": int(score * 100)
                })

            back.append({
                'result': result,
                # optionally you can include prediction scores that you can use to sort the tasks and do active learning
                'score': int(sum(scores) / max(len(scores), 1)),
                'model_version': self.name+"-"+self.version
            })

        response = {
            'results': back,
            'model_version': self.name+"-"+self.version
        }
        # print(response)
        return response

    # web每次用户请求推理，用于对接web界面请求
    # @pysnooper.snoop()
    def inference(self,image,**kwargs):
        save_dir = 'results'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{int(time.time() * 1000)}.jpg')  # img.jpg
        results = self.model(image,device)
        result = None
        if results:
            result = list(results)[0]

        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(save_path)  # save image

        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "image": save_path
            }
        ]
        return back

model=YOLOV8_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(image='client.jpg')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py web --save_model_dir xx
    model.run()

