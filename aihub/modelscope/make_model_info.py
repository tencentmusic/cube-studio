import re
import shutil
from jinja2 import Environment, BaseLoader, DebugUndefined
import requests,json
import os,sys

def make_aihub_info():
    all_info=[]
    import os


    for model_name in os.listdir('.'):
        if model_name in ['app1','demo','aihub','modelscope']:
            continue
        info_path = os.path.join(model_name,'info.json')
        if os.path.exists(info_path):
            info = json.load(open(info_path))
            info['price']="1"
            print(model_name)
            all_info.append(info)
            # print(model_name)

    json.dump(all_info,open('info.json',mode='w'),indent=4,ensure_ascii=False)

make_aihub_info()