
import json,datetime,time,os,sys
import argparse
import time, datetime
import os
import requests
import random
import ray

@ray.remote
def download_file(index,paths):
    print('%s,download array %s,len %s'%(datetime.datetime.now(),index,len(paths)))
    for path in paths:
        local_path, url = path[0], path[1]

        # print(local_path,url)
        try:
            if os.path.exists(local_path):
                continue
                # os.remove(local_path)
            base_dir = os.path.dirname(local_path)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            # cmd = 'wget %s -O %s'%(url,local_path)
            # # print(cmd)
            # os.system(cmd)

            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                continue

            with open(local_path, 'ab') as f:
                f.write(res.content)
                f.flush()
                # 关闭文件
                f.close()

            # # return 0, local_path
            # if not os.path.exists(local_path):
            #     raise 'not exist'
            #     # return
        except Exception as e:
            print('%s error'%path)


def main(src_file_path):
    all_urls_info = open(src_file_path, mode='r').readlines()
    all_urls_info = [url.replace('\n', '').strip() for url in all_urls_info if url.replace('\n', '').strip()]
    print('total num %s' % len(all_urls_info))
    tasks = []
    paths = [[] for i in range(1000)]   # 划分成1000个任务盒
    index=0
    for url_info in all_urls_info:
        one_url_arr=url_info.replace('\t',' ').strip().split(' ')
        one_url_arr=[vid for vid in one_url_arr if vid]

        if len(one_url_arr)>1:
            url,local_path = one_url_arr[0], one_url_arr[1]
            paths[index].append([local_path, url])
            index = (index+1)%1000

    for index,path in enumerate(paths):
        if path:
            tasks.append(download_file.remote(index,path))
    if tasks:
        ray.get(tasks)

    time.sleep(10)



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="build component")
    arg_parser.add_argument('--num_workers', type=int, required=False, help="workers的数量", default=3)
    arg_parser.add_argument('--download_type', type=str, required=False, help="数据下载类型", default="url")
    arg_parser.add_argument('--input_file', type=str, required=False, help="下载内容文件地址", default="/mnt/ray/url.txt")

    args = arg_parser.parse_args()
    print('NUM_WORKER',args.num_workers)

    NUM_WORKER = int(args.num_workers)

    if args.download_type=='url':
        from job.pkgs.k8s.py_ray import ray_launcher
        head_service_ip = ray_launcher(int(args.num_workers), '', 'create')
        print('head_service_ip: ' + head_service_ip)
        if not head_service_ip:
            raise RuntimeError("ray cluster not found")

        main(src_file_path=args.input_file)
        ray_launcher(int(args.num_workers), '', 'delete')



