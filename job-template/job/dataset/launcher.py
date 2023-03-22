
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

host = os.getenv('HOST',os.getenv('KFJ_MODEL_REPO_API_URL','http://kubeflow-dashboard.infra')).strip('/')

@pysnooper.snoop()
def download(**kwargs):
    # print(kwargs)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': KFJ_CREATOR
    }

    # 获取项目组
    url = host + "/dataset_modelview/api/?form_data=" + json.dumps({
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

    # 查询同名是否存在，创建者是不是指定用户
    url = host+"/inferenceservice_modelview/api/?form_data="+json.dumps({
        "filters":[
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
    })

    # print(url)
    res = requests.get(url,headers=headers, allow_redirects=False)
    # print(res.content)
    if res.status_code==200:

        payload = {
            'model_name': kwargs['model_name'],
            'model_version': kwargs['model_version'],
            'model_path':kwargs['model_path'],
            'label': kwargs['label'],
            'project': exist_project['id'],
            'images': kwargs['images'],
            'working_dir': kwargs['working_dir'],
            'command': kwargs['command'],
            'args': kwargs['args'],
            'env': kwargs['env'],
            'resource_memory': kwargs['resource_memory'],
            'resource_cpu': kwargs['resource_cpu'],
            'resource_gpu': kwargs['resource_gpu'],
            'min_replicas': kwargs['replicas'],
            'max_replicas': kwargs['replicas'],
            'ports': kwargs['ports'],
            'volume_mount': kwargs['volume_mount'],
            'host': kwargs['host'],
            'hpa': kwargs['hpa'],
            'service_type':kwargs['service_type']
        }

        exist_services = res.json().get('result',{}).get('data',[])
        new_service=None
        # 不存在就创建新的服务
        if not exist_services:
            url = host + "/inferenceservice_modelview/api/"
            res = requests.post(url, headers=headers,json=payload, allow_redirects=False)
            if res.status_code==200:
                new_service = res.json().get('result', {})
            # print(res)

        else:
            exist_service=exist_services[0]
            # 更新服务
            url = host + "/inferenceservice_modelview/api/%s"%exist_service['id']
            res = requests.put(url, headers=headers, json=payload, allow_redirects=False)
            if res.status_code==200:
                new_service = res.json().get('result',{})
            # print(res)

        if new_service:
            time.sleep(5)  # 等待数据刷入数据库
            print(new_service)
            url = host + "/inferenceservice_modelview/deploy/prod/%s"%new_service['id']
            res = requests.get(url,headers=headers, allow_redirects=False)
            if res.status_code==302 or res.status_code==200:
                print('部署成功')
            else:
                print(res.content)
                print('部署失败')
                exit(1)

    else:
        print(res.content)
        exit(1)



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("deploy service launcher")
    arg_parser.add_argument('--src_type', type=str, help="数据集来源", default='当前平台')
    arg_parser.add_argument('--name', type=str, help="数据集名称", default='')
    arg_parser.add_argument('--version', type=str, help="数据集版本", default='')
    arg_parser.add_argument('--partition', type=str, help="数据集分区", default='')
    arg_parser.add_argument('--save_dir', type=str, help="保存目录", default='')

    args = arg_parser.parse_args()
    # print("{} args: {}".format(__file__, args))
    if args.src_type=='当前平台':
        download(**args.__dict__)
    elif args.src_type=='hugging-face':
        pass
    elif args.src_type=='魔塔':
        pass


