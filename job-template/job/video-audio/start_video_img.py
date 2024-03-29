
import ray
import re

import os
import sys
import time
from kubernetes import client, config, watch

sys.path.append(os.path.dirname(__file__))

import argparse
import datetime, time
import pysnooper

import time,datetime
import os
import sys
import requests
import random
import json
import traceback
import subprocess
import shutil
import ray

@ray.remote
def video_to_images(index,paths):
    print('%s, video to images %s,len %s' % (datetime.datetime.now(), index, len(paths)))
    for path in paths:
        try:
            local_path, des_dir, frame_rate = path[0],path[1],path[2]

            # if os.path.exists(des_dir):
            #     print('path %s exist'%des_dir)
            #     continue

            base_dir = os.path.dirname(des_dir)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)


            cmd = 'ffmpeg -i {} -r {} -f image2 {}/%5d.jpg'.format(local_path, frame_rate, des_dir)
            child_proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            result = child_proc.communicate()
            out = eval(result[0])
            # print(out)

        except:
            print('%s error'%path)


def main(src_file_path):
    all_videos_info = open(src_file_path, mode='r').readlines()
    all_videos_info = [video_info.replace('\n', '').strip() for video_info in all_videos_info if video_info.replace('\n', '').strip()]
    print('total num %s'%len(all_videos_info))
    tasks = []
    paths = [[] for i in range(1000)]  # 划分成1000个任务盒
    index = 0
    for video_info in all_videos_info:
        one_video_arr=video_info.replace('\t',' ').strip().split(' ')
        one_video_arr=[video for video in one_video_arr if video]

        if len(one_video_arr)>2:
            local_path,des_dir,frame_rate = one_video_arr[0], one_video_arr[1], one_video_arr[2]
            paths[index].append([local_path, des_dir,frame_rate])
            index = (index + 1) % 1000

    for index, path in enumerate(paths):
        if path:
            tasks.append(video_to_images.remote(index, path))

    if tasks:
        ray.get(tasks)



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="build component")
    arg_parser.add_argument('--num_workers', type=int, required=False, help="workers的数量", default=3)
    arg_parser.add_argument('--input_file', type=str, required=False, help="配置文件地址", default="/mnt/ray/url.txt")

    args = arg_parser.parse_args()
    print('NUM_WORKER',args.num_workers)

    NUM_WORKER = int(args.num_workers)

    from job.pkgs.k8s.py_ray import ray_launcher

    head_service_ip = ray_launcher(int(args.num_workers), '', 'create')
    print('head_service_ip: ' + head_service_ip)
    if not head_service_ip:
        raise RuntimeError("ray cluster not found")

    main(src_file_path=args.input_file)
    ray_launcher(int(args.num_workers), '', 'delete')
