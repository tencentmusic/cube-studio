
import os,sys
import argparse
import datetime
import json
import time
from multiprocessing import Pool
from functools import partial
import uuid
import pysnooper
import re
import requests
import copy
import os
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME','public')

host = os.getenv('HOST',os.getenv('KFJ_MODEL_REPO_API_URL','http://kubeflow-dashboard.infra')).strip('/')

# @pysnooper.snoop()
def download_file(url,des_dir=None,local_path=None):
    if des_dir:
        local_path = os.path.join(des_dir, url.split('/')[-1])
    print(f'begin donwload {local_path} from {url}')
    # 注意传入参数 stream=True
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        if os.path.exists(local_path):
            print(local_path,'已经存在')
            return
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=102400):
                f.write(chunk)
        r.close()

@pysnooper.snoop()
def download(name,version,partition,save_dir,**kwargs):
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
                "value":name
            },
            {
                "col": "version",
                "opr": "eq",
                "value": version
            }
        ]
    })
    res = requests.get(url, headers=headers)
    exist_dataset = res.json().get('result', {}).get('data', [])
    if not exist_dataset:
        print('不存在指定数据集或指定版本')
        exit(1)
    exist_dataset = exist_dataset[0]

    # 查询同名是否存在，创建者是不是指定用户
    url = host+f"/dataset_modelview/api/download/{exist_dataset['id']}"
    if partition:
        url = url + "/" + partition
    # print(url)
    res = requests.get(url,headers=headers, allow_redirects=False)
    # print(res.content)
    if res.status_code==200:
        donwload_urls = res.json().get("result", {}).get("download_urls", [])
        print(donwload_urls)
        os.makedirs(save_dir, exist_ok=True)
        pool = Pool(len(donwload_urls))  # 开辟包含指定数目线程的线程池
        pool.map(partial(download_file, des_dir=save_dir), donwload_urls)  # 当前worker，只处理分配给当前worker的任务
        pool.close()
        pool.join()

        # 对目录下的压缩文件进行解压
        files = os.listdir(save_dir)
        for file in files:
            try:
                if '.zip' in file:
                    exe_command(f'cd {save_dir} && unzip {file}')
                elif '.tar.gz' in file:
                    exe_command(f'cd {save_dir} && tar -zxvf {file}')
                elif '.gz' in file:
                    exe_command(f'cd {save_dir} && gzip -d {file}')
            except Exception as e:
                print(e)
        exit(0)

    exit(1)

from subprocess import Popen, PIPE, STDOUT

def exe_command(command):
    """
    执行 shell 命令并实时打印输出
    :param command: shell 命令
    :return: process, exitcode
    """
    print(command)
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip(),flush=True)
    exitcode = process.wait()
    return exitcode

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("download dataset launcher")
    arg_parser.add_argument('--src_type', type=str, help="数据集来源", default='当前平台')
    arg_parser.add_argument('--name', type=str, help="数据集名称", default='')
    arg_parser.add_argument('--version', type=str, help="数据集版本", default='latest')
    arg_parser.add_argument('--partition', type=str, help="数据集分区", default='')
    arg_parser.add_argument('--save_dir', type=str, help="保存目录", default='')

    args = arg_parser.parse_args()
    if not args.save_dir:
        args.save_dir = f'/mnt/{KFJ_CREATOR}/dataset/{args.name}/{args.version}'
        if args.partition:
            args.save_dir=f'/mnt/{KFJ_CREATOR}/dataset/{args.name}/{args.version}/{args.partition}'
    # print("{} args: {}".format(__file__, args))
    if args.src_type=='cube-studio' or args.src_type=='当前平台':
        download(**args.__dict__)
    elif args.src_type=='huggingface':
        command = f'huggingface-cli download --repo-type dataset --resume-download {args.name} --revision {args.version} --local-dir {args.save_dir} --local-dir-use-symlinks False'
        exitcode = exe_command(command)
        exit(exitcode)

    elif args.src_type=='modelscope':
            pass

