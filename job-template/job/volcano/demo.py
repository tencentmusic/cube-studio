import time, datetime, json, requests, io, os
from multiprocessing import Pool
from functools import partial
import os, random, sys

WORLD_SIZE = int(os.getenv('VC_WORKER_NUM', '1'))  # 总worker的数目
RANK = int(os.getenv("VC_TASK_INDEX", '0'))  # 当前是第几个worker 从0开始

print(WORLD_SIZE, RANK)


# 子进程要执行的代码
def task(key):
    print(datetime.datetime.now(),'worker:', RANK, ', task:', key, flush=True)
    time.sleep(1)


if __name__ == '__main__':
    # if os.path.exists('./success%s' % RANK):
    #     os.remove('./success%s' % RANK)

    input = range(300)  # 所有要处理的数据
    local_task = []  # 当前worker需要处理的任务
    for index in input:
        if index % WORLD_SIZE == RANK:
            local_task.append(index)  # 要处理的数据均匀分配到每个worker

    # 每个worker内部还可以用多进程，线程池之类的并发操作。
    pool = Pool(10)  # 开辟包含指定数目线程的线程池
    pool.map(partial(task), local_task)  # 当前worker，只处理分配给当前worker的任务
    pool.close()
    pool.join()

    # 添加文件标识，当前worker结束
    # open('./success%s' % RANK, mode='w').close()
    # # rank0做聚合操作
    # while (RANK == 0):
    #     success = [x for x in range(WORLD_SIZE) if os.path.exists('./success%s' % x)]
    #     if len(success) != WORLD_SIZE:
    #         time.sleep(5)
    #     else:
    #         # 所有worker全部结束，worker0开始聚合操作
    #         print('begin reduce')
    #         break