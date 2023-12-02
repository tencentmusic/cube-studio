
import os,sys
import argparse
import datetime
import json
import time
import uuid
import pysnooper
import re
import requests
import copy
import os
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME','public')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')
KFJ_PIPELINE_ID = os.getenv('KFJ_PIPELINE_ID', '0')
host = os.getenv('HOST',os.getenv('KFJ_MODEL_REPO_API_URL','http://kubeflow-dashboard.infra')).strip('/')

# @pysnooper.snoop()
def deploy(**kwargs):
    # print(kwargs)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': KFJ_CREATOR
    }

    # 获取项目组
    url = host + "/project_modelview/api/?form_data=" + json.dumps({
        "filters": [
            {
                "col": "name",
                "opr": "eq",
                "value": kwargs['project_name']
            }
        ]
    })
    res = requests.get(url, headers=headers)
    exist_project = res.json().get('result', {}).get('data', [])
    if not exist_project:
        print('不存在项目组')
        return
    exist_project = exist_project[0]

    # 查询同名，同run-id是否存在
    url = host+"/training_model_modelview/api/?form_data="+json.dumps({
        "filters":[
            {
                "col": "name",
                "opr": "eq",
                "value": kwargs['model_name']
            },
            {
                "col": "run_id",
                "opr": "eq",
                "value": KFJ_RUN_ID
            }
        ]
    })

    # print(url)
    res = requests.get(url,headers=headers, allow_redirects=False)
    # print(res.content)
    if res.status_code==200:

        payload = {
            'name': kwargs['model_name'],
            'version': kwargs['model_version'],
            'path':kwargs['model_path'],
            'describe': kwargs['describe'],
            'project': exist_project['id'],
            'framework': kwargs['framework'],
            'run_id': KFJ_RUN_ID,
            'run_time': datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
            'metrics': kwargs['model_metric'],
            'md5': '',
            'api_type': kwargs['inference_framework'],
            'pipeline_id': KFJ_PIPELINE_ID
        }

        exist_models = res.json().get('result',{}).get('data',[])

        # 不存在就创建新的模型
        if not exist_models:
            url = host + "/training_model_modelview/api/"
            res = requests.post(url, headers=headers,json=payload, allow_redirects=False)
            if res.status_code==200:
                new_model = res.json().get('result', {})
                print('注册成功')
            else:
                print(res.content)
                print('注册失败')
                exit(1)

            # print(res)

        else:
            exist_model=exist_models[0]
            # 更新模型
            url = host + "/training_model_modelview/api/%s"%exist_model['id']
            res = requests.put(url, headers=headers, json=payload, allow_redirects=False)
            if res.status_code==200:
                new_model = res.json().get('result',{})
                print('更新成功')
            else:
                print(res.content)
                print('更新失败')
                exit(1)


    else:
        print(res.content)
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("deploy service launcher")
    arg_parser.add_argument('--project_name', type=str, help="所属项目组", default='public')
    arg_parser.add_argument('--model_name', type=str, help="模型名", default='demo')
    arg_parser.add_argument('--model_version', type=str, help="模型版本号", default=datetime.datetime.now().strftime('v%Y.%m.%d.1'))
    arg_parser.add_argument('--model_path', type=str, help="模型地址", default='')
    arg_parser.add_argument('--model_metric', type=str, help="模型指标", default='')
    arg_parser.add_argument('--describe', type=str, help="模型描述", default='xx模型')
    arg_parser.add_argument('--framework', type=str, help="算法框架", default='tf')
    arg_parser.add_argument('--inference_framework', type=str, help="推理框架", default='tfserving')

    args = arg_parser.parse_args()
    # print("{} args: {}".format(__file__, args))

    deploy(**args.__dict__)


