
import sys, os
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
dir_common = os.path.split(os.path.realpath(__file__))[0] + '/../'
print(dir_common)
sys.path.append(dir_common)  # 将根目录添加到系统目录,才能正常引用common文件夹

import datetime
import redis

import logging
import time
import pysnooper
import json
import requests
from project import push_admin


@pysnooper.snoop()
def check_push():
    try:
        redis_host = os.getenv('REDIS_HOST', 'redis-master.infra')
        r = redis.StrictRedis(host=redis_host,
                              port=6379,
                              db=0,
                              decode_responses=True,
                              password='admin')
        # r = redis.StrictRedis(host='100.116.64.86', port=8080, db=0, decode_responses=True, password='admin')
        # r = redis.StrictRedis(host='9.22.26.233', port=8080, db=0, decode_responses=True, password='admin')

        if r.exists('celery'):
            unscheduld_num = r.llen('celery')
            print(unscheduld_num)
            if unscheduld_num > 100:
                push_admin(__('超过100个任务堆积未被调度'))
                return
        if r.exists('unacked'):
            unacked_num = r.hlen('unacked')
            print(unacked_num)
            if unacked_num > 500:
                push_admin(__("超过500个调度未完成"))

    except Exception as e:
        print(e)
        push_admin(str(e))


import argparse

if __name__ == '__main__':
    check_push()
