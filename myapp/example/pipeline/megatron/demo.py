import time, datetime, json, requests, io, os
from multiprocessing import Pool
from functools import partial
import os, random, sys

ROLE = os.getenv('VC_TASK_ROLE','worker')
HOST = os.getenv(f'VC_{ROLE.upper()}_HOSTS','')
WORLD_SIZE = int(os.getenv(f'VC_{ROLE.upper()}_NUM', '1'))  # 总worker的数目
RANK = int(os.getenv("VC_TASK_INDEX", '0'))  # 当前是第几个worker 从0开始
GPU_NUM = int(os.getenv("GPU_NUM", '0'))  # 当前是第几个worker 从0开始
print('ROLE',ROLE,flush=True)
print('HOST',HOST,flush=True)
print('WORLD_SIZE',WORLD_SIZE,flush=True)
print("RANK",RANK,flush=True)
print('GPU_NUM',GPU_NUM,flush=True)
# todo 每个角色，要干的事情
time.sleep(100)