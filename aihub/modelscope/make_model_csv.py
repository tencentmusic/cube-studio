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
        model_name = model.get("Name", '').replace('.', '_').replace('_', "-").lower()
        if os.path.exists(model_name):
            file.write("'%s','%s','%s'\n"%(model_name,model['ChineseName'],model['Description']))

    file.close()

make_aihub()