
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
def video_to_audio(index,paths):
    print('%s, video to audio %s,len %s' % (datetime.datetime.now(), index, len(paths)))
    for path in paths:
        try:
            local_path, des_path = path[0], path[1]
            if os.path.exists(des_path):
                print('%s exist'%des_path)
                continue
            base_dir = os.path.dirname(des_path)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            if '.mp3' in des_path:
                cmd_video2mp3 = 'ffmpeg -loglevel quiet -i {} -y {}'.format(local_path, des_path)
                child_proc = subprocess.Popen(cmd_video2mp3, shell=True, stdout=subprocess.PIPE)
                result = child_proc.communicate()

            if '.wav' in des_path:
                mp3_path=des_path.replace('.wav','.mp3')
                cmd_video2mp3 = 'ffmpeg -loglevel quiet -i {} -y {}'.format(local_path, mp3_path)
                child_proc = subprocess.Popen(cmd_video2mp3, shell=True, stdout=subprocess.PIPE)
                result = child_proc.communicate()

                cmd_decode = 'ffmpeg -loglevel quiet -i %s -map_channel 0.0.1 -ar 16000 -acodec pcm_s16le -y %s'%(mp3_path, des_path)
                child_proc = subprocess.Popen(cmd_decode, shell=True, stdout=subprocess.PIPE)
                result = child_proc.communicate()

                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
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

        if len(one_video_arr)>1:
            local_path,des_path = one_video_arr[0], one_video_arr[1]
            paths[index].append([local_path, des_path])
            index = (index + 1) % 1000

    for index, path in enumerate(paths):
        if path:
            tasks.append(video_to_audio.remote(index, path))

    if tasks:
        ray.get(tasks)

    time.sleep(10)
