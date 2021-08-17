

from urllib.parse import urlparse

import prometheus_client
from prometheus_client.core import CollectorRegistry
from prometheus_client import CollectorRegistry, Gauge
import base64
import json,datetime,time
import logging
import os
import io
import requests
import pysnooper

class Prometheus():

    def __init__(self,host=''):
        self.host = host


    # @pysnooper.snoop()
    def get_metric(self,pod_name, namespace):
        print(self.host)
        max_cpu = 0
        max_mem = 0

        # 这个pod  30分钟内的最大值
        mem_expr = "sum by (pod) (container_memory_usage_bytes{namespace='%s', pod=~'%s.*',container!='POD', container!=''})"%(namespace,pod_name)
        # print(mem_expr)
        params={
            'query': mem_expr,
            'start':(datetime.datetime.now()-datetime.timedelta(days=1)-datetime.timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'end':datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'step':"1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        print(params)

        try:
            res = requests.get(url=self.host,params=params)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status']=='success':
                metrics=metrics['data']['result']
                if metrics:
                    metrics=metrics[0]['values']
                    for metric in metrics:
                        if int(metric[1])>max_mem:
                            max_mem = int(metric[1])/1024/1024/1024

        except Exception as e:
            print(e)
            return {}


        cpu_expr = "sum by (pod) (rate(container_cpu_usage_seconds_total{namespace='%s',pod=~'%s.*',container!='POD'}[1m]))" % (namespace, pod_name)

        params={
            'query': cpu_expr,
            'start':(datetime.datetime.now()-datetime.timedelta(days=1)-datetime.timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'end':datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'step':"1m",   # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        print(params)
        try:

            res = requests.get(url=self.host,params=params)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status']=='success':
                metrics=metrics['data']['result']
                if metrics:
                    metrics=metrics[0]['values']
                    for metric in metrics:
                        if float(metric[1])>max_cpu:
                            max_cpu = float(metric[1])
        except Exception as e:
            print(e)
            return {}

        return {"cpu":round(max_cpu, 2),"memory":round(max_mem,2)}



# if __name__ == "__main__":
#     prometheus = Prometheus('http://9.138.244.68:8080/api/v1/query_range')
#     prometheus.get_metric('cupid-deepfm-v1-x7g9h-515457540','pipeline')






