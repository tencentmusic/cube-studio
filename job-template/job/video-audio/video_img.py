
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

