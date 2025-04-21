# ===================用户代码=====================
import json
import os
import numpy
import requests
from predict_model import Offline_Predict  # 需要引入这个包里面的类
import time
import pysnooper
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


class My_Offline_Predict(Offline_Predict):

    # @pysnooper.snoop()
    def __init__(self):
        # 加载预训练的 ResNet-50 模型
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load('resnet50.pth'))
        # 切换到评估模式
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 定义所有要处理的数据源，返回字符串列表
    def datasource(self):
        all_lines = open('images_url.txt', mode='r').readlines()
        return all_lines

    # 定义一条数据的处理逻辑
    # @pysnooper.snoop()
    def predict(self, value):
        os.makedirs('download', exist_ok=True)
        # 加载图像并在 GPU 上进行推理
        image_path = value

        if 'https://' in value or 'http://' in value:
            file_name = value[value.rindex("/") + 1:]
            image_path = f'download/{file_name}'
            file = open(image_path, mode='wb')
            file.write(requests.get(value).content)
            file.close()
        time.sleep(1)
        # image = Image.open(image_path).convert('RGB')
        #
        # image_tensor = self.transform(image).unsqueeze(0)
        # with torch.no_grad():
        #     outputs = self.model(image_tensor)
        #     _, preds = torch.max(outputs, 1)
        # result = preds.item()
        # print(value)
        # print(result)
        # exist_result={}
        # if os.path.exists('result.json'):
        #     exist_result = json.load(open('result.json'))
        # exist_result[value]=result
        # json.dump(exist_result,open('result.json',mode='w'),indent=4,ensure_ascii=False)
        # return result


My_Offline_Predict().run()


