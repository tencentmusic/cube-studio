import re
import shutil
from jinja2 import Environment, BaseLoader, DebugUndefined
import requests,json
import os,sys

def make_aihub():
    all_models = json.load(open("all_models.json", mode='r'))
    file = open('model.csv', mode='w')
    file.write('模型名,中文名,描述\n')
    for model in all_models:
        model_name = model.get("Name", '').replace('_', "-").replace('.', '_')

        # # 还在集成的不需要
        # if model['Integrating']<2:
        #     print(model_path,'集成中')
        #     continue
        # 使用有障碍的不要
        # if model['IsAccessible']==0:
        #     print(model_path, '有障碍')
        #     continue
        # 下载量太少的不要
        # if int(model['Downloads'])<100:
        #     # print(model_path, '下载少')
        #     continue

        if model.get('Path', '')=='damo':
            if os.path.exists(model_name):
                file.write("'%s','%s','%s'\n"%(model_name,model['ChineseName'],model['Description']))

    file.close()

make_aihub()