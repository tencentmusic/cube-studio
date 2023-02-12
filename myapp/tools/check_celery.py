import sys
import os

import datetime
import redis

import logging
import time
import pysnooper
import json
import requests

def push_admin(message):
    pass

@pysnooper.snoop()
def check_push():
    try:
        redis_host = os.getenv('REDIS_HOST', 'redis-master.infra')
        r = redis.StrictRedis(host=redis_host,
                              port=6379,
                              db=0,
                              decode_responses=True,
                              password='admin')

        if r.exists('celery'):
            unscheduld_num = r.llen('celery')
            print(unscheduld_num)
            if unscheduld_num > 100:
                push_admin('超过100个任务堆积未被调度')
                return
        if r.exists('unacked'):
            unacked_num = r.hlen('unacked')
            print(unacked_num)
            if unacked_num > 500:
                push_admin("超过500个调度未完成")

    except Exception as e:
        print(e)
        push_admin(str(e))


import argparse

if __name__ == '__main__':
    check_push()
