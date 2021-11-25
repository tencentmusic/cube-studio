

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
        #  '/api/v1/query_range'    查看范围数据
        #  '/api/v1/query'    瞬时数据查询
        self.host = host
        self.query_path='http://%s/api/v1/query'%self.host
        self.query_range_path = 'http://%s/api/v1/query_range' % self.host


        # @pysnooper.snoop()
    def get_resource_metric(self,pod_name, namespace):
        max_cpu = 0
        max_mem = 0
        ave_gpu = 0

        # 这个pod  30分钟内的最大值
        mem_expr = "sum by (pod) (container_memory_working_set_bytes{namespace='%s', pod=~'%s.*',container!='POD', container!=''})"%(namespace,pod_name)
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
            res = requests.get(url=self.query_range_path,params=params)
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

            res = requests.get(url=self.query_range_path,params=params)
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



        gpu_expr = "avg by (exported_pod) (DCGM_FI_DEV_GPU_UTIL{exported_namespace='%s',exported_pod=~'%s.*'})" % (namespace, pod_name)

        params={
            'query': gpu_expr,
            'start':(datetime.datetime.now()-datetime.timedelta(days=1)-datetime.timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'end':datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'step':"1m",   # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        print(params)
        try:

            res = requests.get(url=self.query_range_path,params=params)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status']=='success':
                metrics=metrics['data']['result']
                if metrics:
                    metrics=metrics[0]['values']
                    all_util = [float(metric[1]) for metric in metrics]
                    ave_gpu = sum(all_util)/len(all_util)
        except Exception as e:
            print(e)

        return {"cpu":round(max_cpu, 2),"memory":round(max_mem,2),'gpu':round(ave_gpu,2)}




    @pysnooper.snoop()
    def get_machine_metric(self):

        # 这个pod  30分钟内的最大值
        metrics={
            "pod_num":"sum(kubelet_running_pod_count)by (node)",
            "request_mem":""
        }
        back = {}
        for metric_name in metrics:
            # print(mem_expr)
            params={
                'query': metrics[metric_name],
                'timeout':"30s"
            }
            print(params)
            back[metric_name]={}

            try:
                res = requests.get(url=self.query_path,params=params)
                metrics = json.loads(res.content.decode('utf8', 'ignore'))
                if metrics['status']=='success':
                    metrics=metrics['data']['result']
                    if metrics:
                        for metric in metrics:
                            node = metric['metric']['node']
                            if ':' in node:
                                node = node[:node.index(':')]
                            value = metric['value'][1]
                            back[metric_name][node]=int(value)


            except Exception as e:
                print(e)

        return back

# if __name__ == "__main__":
#     prometheus = Prometheus('10.101.142.128:8081')
#     prometheus.get_machine_metric()






