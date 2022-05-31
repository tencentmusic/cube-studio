
import os,sys
base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)

import argparse
import datetime
import json
import time
import uuid
import os
import pysnooper
import os,sys
import re
import requests
import psutil
import copy
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
host = 'http://kubeflow-dashboard.infra'
# host = 'http://127.0.0.1:80'
# host = 'http://9.135.92.226'



@pysnooper.snoop()
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
    res = requests.get(url,headers=headers)
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
            res = requests.post(url, headers=headers,json=payload)
            if res.status_code==200:
                new_service = res.json().get('result', {})
            # print(res)

        else:
            exist_service=exist_services[0]
            # 更新服务
            url = host + "/inferenceservice_modelview/api/%s"%exist_service['id']
            res = requests.put(url, headers=headers, json=payload)
            if res.status_code==200:
                new_service = res.json().get('result',{})
            # print(res)

        if new_service:
            print(new_service)
            url = host + "/inferenceservice_modelview/deploy/prod/%s"%new_service['id']
            res = requests.post(url,headers=headers)
            if res.status_code==302 or res.status_code==200:
                print('部署成功')
            else:
                print('部署失败')




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("deploy service launcher")
    arg_parser.add_argument('--project_name', type=str, help="所属项目组", default='public')
    arg_parser.add_argument('--service_type', type=str, help="服务类型", default='service')
    arg_parser.add_argument('--label', type=str, help="服务中文名", default='演示服务')
    arg_parser.add_argument('--model_name', type=str, help="模型名", default='demo')
    arg_parser.add_argument('--model_version', type=str, help="模型版本号", default=datetime.datetime.now().strftime('v%Y.%m.%d.1'))
    arg_parser.add_argument('--model_path', type=str, help="模型地址", default='')
    arg_parser.add_argument('--images', type=str, help="镜像", default='nginx')

    arg_parser.add_argument('--resource_memory', type=str, help="内存", default='2G')
    arg_parser.add_argument('--resource_cpu', type=str, help="cpu", default='2')
    arg_parser.add_argument('--resource_gpu', type=str, help="gpu", default='0')
    arg_parser.add_argument('--replicas', type=str, help="副本数", default='1')
    arg_parser.add_argument('--host', type=str, help="域名", default='')
    arg_parser.add_argument('--command', type=str, help="启动命令", default='')
    arg_parser.add_argument('--args', type=str, help="启动参数", default='')
    arg_parser.add_argument('--working_dir', type=str, help="工作目录", default='')
    arg_parser.add_argument('--env', type=str, help="环境变量", default='')
    arg_parser.add_argument('--hpa', type=str, help="弹性伸缩", default='')
    arg_parser.add_argument('--ports', type=str, help="端口号", default='80')
    arg_parser.add_argument('--volume_mount', type=str, help="挂载", default='')


    args = arg_parser.parse_args()
    # print("{} args: {}".format(__file__, args))

    deploy(**args.__dict__)


