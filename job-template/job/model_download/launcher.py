
import os,sys
import argparse
import datetime
import json
import shutil
import time
import uuid
import pysnooper
import re
import requests
import copy
import os
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
host = os.getenv('HOST',os.getenv('KFJ_MODEL_REPO_API_URL','http://kubeflow-dashboard.infra')).strip('/')

@pysnooper.snoop()
def download(**kwargs):
    # print(kwargs)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': KFJ_CREATOR
    }
    model_path=""
    # 从注册的模型中下载模型
    if kwargs['from']=='模型管理':
        url = host + "/training_model_modelview/api/?form_data=" + json.dumps({
            "filters": [
                {
                    "col": "name",
                    "opr": "eq",
                    "value": kwargs['model_name']
                },
                {
                    "col": "version",
                    "opr": "eq",
                    "value": kwargs['model_version']
                }
            ]
        })

        # print(url)
        res = requests.get(url, headers=headers, allow_redirects=False)
        # print(res.content)
        if res.status_code == 200:
            exist_model = res.json().get('result', {}).get('data', [])
            if exist_model:
                exist_model = exist_model[0]
                print(exist_model)
                if exist_model['path']:
                    model_path = exist_model['path']

    elif kwargs['from']=='推理服务':
        filters = [
            {
                "col": "model_name",
                "opr": "eq",
                "value": kwargs['model_name']
            },
            {
                "col": "model_version",
                "opr": "eq",
                "value": kwargs['model_version']
            }
        ]
        if kwargs['model_status']:
            filters.append({
                "col": "model_status",
                "opr": "eq",
                "value": kwargs['model_status']
            })


        url = host+"/inferenceservice_modelview/api/?form_data="+json.dumps({
            "filters":filters
        })

        # print(url)
        res = requests.get(url,headers=headers, allow_redirects=False)
        # print(res.content)
        if res.status_code==200:
            exist_service = res.json().get('result', {}).get('data', [])
            if exist_service:
                exist_service = exist_service[0]
                print(exist_service)
                if exist_service['model_path']:
                    model_path = exist_service['model_path']

    if model_path:
        os.makedirs(model_path,exist_ok=True)
        shutil.copy2(model_path,kwargs['save_path'])
    else:
        print('未发现模型')
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("download model launcher")
    arg_parser.add_argument('--from', type=str, help="模型来源地", default='train_model')
    arg_parser.add_argument('--model_name', type=str, help="模型名", default='demo')
    arg_parser.add_argument('--model_version', type=str, help="模型版本号",default=datetime.datetime.now().strftime('v%Y.%m.%d.1'))
    arg_parser.add_argument('--model_status', type=str, help="模型状态", default='')
    arg_parser.add_argument('--save_path', type=str, help="下载目录", default='')

    args = arg_parser.parse_args()
    # print("{} args: {}".format(__file__, args))

    download(**args.__dict__)


