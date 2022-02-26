
import time, datetime, json, requests, io, os
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import os, random

WORLD_SIZE = int(os.getenv('VC_WORKER_NUM', ''))  # 总worker的数目
RANK = int(os.getenv("VC_TASK_INDEX", '0'))     # 当前是第几个worker 从0开始
print(WORLD_SIZE, RANK)


# 子进程要执行的代码
def task(key):
    print('worker:',RANK,', task:',key,flush=True)
    time.sleep(1)


if __name__ == '__main__':

    input = range(30000)  # 所有要处理的数据
    local_task = []   # 当前worker需要处理的任务
    for index in input:
        if len(input)//WORLD_SIZE*(RANK+1) > index >= len(input)//WORLD_SIZE*RANK:
            local_task.append(index)  # 要处理的数据均匀分配到每个worker

    input = []
    pool = ThreadPool(10)  # 开辟包含指定数目线程的线程池
    pool.map(partial(task), local_task)  # 当前worker，只处理分配给当前worker的任务
    pool.close()
    pool.join()


