import re
import shutil
from jinja2 import Environment, BaseLoader, DebugUndefined
import requests,json
import os,sys,cv2

def make_aihub_info():
    all_info=[]
    import os
    all_models = json.load(open("all_models.json", mode='r'))
    for model in all_models:
        model_name = model.get("Name", '').replace('.', '_').replace('_', "-").lower()
        if os.path.exists(model_name):
            info={
                "price":"1",
                "name":model_name,
                "label":model['ChineseName'],
                "describe":model['Description'],
                "hot":model['Downloads'],
                "pic":"example.jpg",
                "uuid":model_name
            }
            field = ' '.join(model['Domain'])
            field = '自然语言' if 'nlp' in field else '机器视觉' if 'cv' in field else '听觉' if 'audio' in field else '多模态' if 'multi-modal' in field else "未知"
            info['field'] = field
            if os.path.exists(os.path.join(model_name,'example.jpeg')):
                info['pic'] = "example.jpeg"
            if os.path.exists(os.path.join(model_name,'example.png')):
                info['pic'] = "example.png"
            if os.path.exists(os.path.join(model_name,'example.gif')):
                info['pic'] = "example.gif"
            all_info.append(info)
            # print(model_name)

    json.dump(all_info,open('info.json',mode='w'),indent=4,ensure_ascii=False)

make_aihub_info()