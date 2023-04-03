
import math
from flask import Markup
from jinja2 import Environment, BaseLoader, DebugUndefined
from myapp import app, appbuilder,db
from flask import request
from .baseFormApi import (
    MyappFormRestApi
)

import datetime,time,json

from ..utils.py.py_k8s import K8s

conf = app.config
logging = app.logger


pipeline_resource_used={
    "check_time": None,
    "data": {}
}
node_resource_used = {
    "check_time": None,
    "data": {}
}
global_cluster_load = {}

# 机器学习首页资源弹窗
# @pysnooper.snoop()
def node_traffic():
    if not node_resource_used['check_time'] or node_resource_used['check_time'] < (datetime.datetime.now() - datetime.timedelta(minutes=10)):

        all_node_json={}
        clusters = conf.get('CLUSTERS', {})
        for cluster_name in clusters:
            try:
                cluster = clusters[cluster_name]
                k8s_client = K8s(cluster.get('KUBECONFIG', ''))

                all_node = k8s_client.get_node()
                all_node_resource = k8s_client.get_all_node_allocated_resources()
                all_node_json[cluster_name]={}
                for node in all_node:
                    all_node_json[cluster_name][node['hostip']]=node
                    node_allocated_resources=all_node_resource.get(node['name'],{
                        "used_cpu":0,
                        "used_memory":0,
                        "used_gpu":0
                    })
                    all_node_json[cluster_name][node['hostip']].update(node_allocated_resources)
            except Exception as e:
                print(e)

        node_resource_used['data']=all_node_json
        node_resource_used['check_time'] = datetime.datetime.now()

    all_node_json = node_resource_used['data']
    # 数据格式说明 dict:
    # 'delay': Integer 延时隐藏 单位: 毫秒 0为不隐藏
    # 'hit': Boolean 是否命中
    # 'target': String 当前目标
    # 'type': String 类型 目前仅支持html类型
    # 'title': String 标题
    # 'content': String 内容html内容
    # /static/appbuilder/mnt/make_pipeline.mp4
    message = ''
    td_html = '<td style="border: 1px solid black;padding: 10px">%s</th>'
    message += "<tr>%s %s %s %s %s %s %s<tr>" % (
        td_html % "集群", td_html % "资源组", td_html % "机器", td_html % "机型", td_html % "cpu占用率", td_html % "内存占用率",
        td_html % "gpu占用率")

    global global_cluster_load
    for cluster_name in all_node_json:
        global_cluster_load[cluster_name] = {
            "cpu_req": 0,
            "cpu_all": 0,
            "mem_req": 0,
            "mem_all": 0,
            "gpu_req": 0,
            "gpu_all": 0
        }
        nodes = all_node_json[cluster_name]
        # nodes = sorted(nodes.items(), key=lambda item: item[1]['labels'].get('org','public'))
        # ips = [node[0] for node in nodes]
        # values = [node[1] for node in nodes]
        # nodes = dict(zip(ips,values))

        # 按项目组和设备类型分组
        stored_nodes = {}
        for ip in nodes:
            org = nodes[ip]['labels'].get('org', 'public')
            device = 'cpu'
            if 'gpu' in nodes[ip]['labels']:
                device = 'gpu/' + nodes[ip]['labels'].get('gpu-type', '')
            if 'vgpu' in nodes[ip]['labels']:
                device = 'vgpu/' + nodes[ip]['labels'].get('gpu-type', '')
            if org not in stored_nodes:
                stored_nodes[org] = {}
            if device not in stored_nodes[org]:
                stored_nodes[org][device] = {}
            stored_nodes[org][device][ip] = nodes[ip]
        nodes = {}
        for org in stored_nodes:
            for device in stored_nodes[org]:
                nodes.update(stored_nodes[org][device])

        cluster_config = conf.get('CLUSTERS', {}).get(cluster_name, {})
        grafana_url = "http://"+cluster_config.get('HOST', request.host) + conf.get('GRAFANA_CLUSTER_PATH')
        for ip in nodes:
            node_dashboard_url = "http://" + cluster_config.get('HOST', request.host) + conf.get('K8S_DASHBOARD_CLUSTER') + '#/node/%s?namespace=default' % nodes[ip]['name']
            org = nodes[ip]['labels'].get('org', 'public')
            enable_train = nodes[ip]['labels'].get('train', 'true')
            ip_html = '<a target="_blank" href="%s">%s</a>' % (node_dashboard_url, ip)
            share = nodes[ip]['labels'].get('share', 'true')
            clolr = "#FFFFFF" if share == 'true' else '#F0F0F0'
            device = 'cpu'
            if 'gpu' in nodes[ip]['labels']:
                device = 'gpu/' + nodes[ip]['labels'].get('gpu-type', '')
            if 'vgpu' in nodes[ip]['labels']:
                device = 'vgpu/' + nodes[ip]['labels'].get('gpu-type', '')

            message += '<tr bgcolor="%s">%s %s %s %s %s %s %s<tr>' % (
                clolr,
                td_html % cluster_name,
                td_html % org,
                td_html % ip_html,
                td_html % ('<a target="blank" href="%s">%s</a>' % (grafana_url, device)),
                td_html % ("cpu:%s/%s" % (nodes[ip]['used_cpu'], nodes[ip]['cpu'])),
                td_html % ("mem:%s/%s" % (nodes[ip]['used_memory'], nodes[ip]['memory'])),
                td_html % ("gpu:%s/%s" % (round(nodes[ip]['used_gpu'],2) if 'vgpu' in device else int(float(nodes[ip]['used_gpu'])), nodes[ip]['gpu'])),
                # td_html % (','.join(list(set(nodes[ip]['user']))[0:1]))
            )

            global_cluster_load[cluster_name]['cpu_req'] += int(nodes[ip]['used_cpu'])
            global_cluster_load[cluster_name]['cpu_all'] += int(nodes[ip]['cpu'])
            global_cluster_load[cluster_name]['mem_req'] += int(nodes[ip]['used_memory'])
            global_cluster_load[cluster_name]['mem_all'] += int(nodes[ip]['memory'])
            global_cluster_load[cluster_name]['gpu_req'] += round(float(nodes[ip]['used_gpu']),2)
            global_cluster_load[cluster_name]['gpu_all'] += int(float(nodes[ip]['gpu']))

    message = Markup('<table style="margin:20px">%s</table>' % message)

    data = {
        'content': message,
        'delay': 300000,
        'hit': True,
        'target': conf.get('MODEL_URLS', {}).get('total_resource', ''),
        'title': '机器负载',
        'style': {
            'height': '600px'
        },
        'type': 'html',
    }
    # 返回模板
    return data

# pipeline每个任务的资源占用情况
# @pysnooper.snoop()
def pod_resource():
    if not pipeline_resource_used['check_time'] or pipeline_resource_used['check_time'] < (datetime.datetime.now() - datetime.timedelta(minutes=10)):
        clusters = conf.get('CLUSTERS', {})
        all_tasks_json = {}
        for cluster_name in clusters:
            cluster = clusters[cluster_name]
            k8s_client = K8s(cluster.get('KUBECONFIG', ''))
            try:
                # 获取pod的资源占用
                all_tasks_json[cluster_name] = {}
                # print(all_node_json)
                for namespace in ['pipeline', 'automl', 'service']:
                    all_tasks_json[cluster_name][namespace] = {}
                    all_pods = k8s_client.get_pods(namespace=namespace)
                    for pod in all_pods:
                        org = pod['node_selector'].get("org", 'public')
                        if org not in all_tasks_json[cluster_name][namespace]:
                            all_tasks_json[cluster_name][namespace][org] = {}
                        if pod['status'] == 'Running':
                            user = pod['labels'].get('user', pod['labels'].get('username', pod['labels'].get('run-rtx',pod['labels'].get('rtx-user',''))))
                            if user:
                                all_tasks_json[cluster_name][namespace][org][pod['name']] = {}
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['username'] = user
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['host_ip'] = pod['host_ip']
                                # print(namespace,pod)
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['request_memory'] = pod['memory']
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['request_cpu'] = pod['cpu']
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['request_gpu'] = pod['gpu']
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_memory'] = '0'
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_cpu'] = '0'
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_gpu'] = '0'

                                # print(namespace,org,pod['name'])

                    # 获取pod的资源使用
                    all_pods_metrics = k8s_client.get_pod_metrics(namespace=namespace)
                    # print(all_pods_metrics)
                    for pod in all_pods_metrics:
                        for org in all_tasks_json[cluster_name][namespace]:
                            if pod['name'] in all_tasks_json[cluster_name][namespace][org]:
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_memory'] = pod['memory']
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_cpu'] = pod['cpu']
                                # print(namespace,org,pod['name'])
                                break
                    # print(all_tasks_json)
            except Exception as e:
                print(e)
        pipeline_resource_used['data'] = all_tasks_json
        pipeline_resource_used['check_time'] = datetime.datetime.now()

    all_tasks_json = pipeline_resource_used['data']
    all_pod_resource=[]
    for cluster_name in all_tasks_json:
        cluster_config = conf.get('CLUSTERS', {}).get(cluster_name, {})
        for namespace in all_tasks_json[cluster_name]:
            for org in all_tasks_json[cluster_name][namespace]:
                for pod_name in all_tasks_json[cluster_name][namespace][org]:
                    pod = all_tasks_json[cluster_name][namespace][org][pod_name]
                    dashboard_url = "http://" + cluster_config.get('HOST', request.host) + conf.get('K8S_DASHBOARD_CLUSTER') + '#/search?namespace=%s&q=%s' % (namespace, pod_name)
                    task_grafana_url = "http://"+cluster_config.get('HOST', request.host) + conf.get('GRAFANA_TASK_PATH')
                    node_grafana_url = "http://"+cluster_config.get('HOST', request.host) + conf.get('GRAFANA_NODE_PATH')

                    pod_resource={
                        "cluster":cluster_name,
                        "project":org,
                        "namespace":Markup('<a target="blank" href="%s">%s</a>' % (dashboard_url, namespace)),
                        "pod":Markup('<a target="blank" href="%s">%s</a>' % (task_grafana_url + pod_name, pod_name)),
                        "username":pod['username'],
                        "node":Markup('<a target="blank" href="%s">%s</a>' % (node_grafana_url + pod["host_ip"], pod["host_ip"])),
                        "cpu":"%s/%s" % (math.ceil(int(pod.get('used_cpu', '0')) / 1000), int(pod.get('request_cpu', '0'))),
                        "memory":"%s/%s" % (int(pod.get('used_memory', '0')), int(pod.get('request_memory', '0'))),
                        "gpu":"%s" % str(round(float(pod.get('request_gpu', '0')),2)),
                    }
                    all_pod_resource.append(pod_resource)
    return all_pod_resource

# 添加api
class Total_Resource_ModelView_Api(MyappFormRestApi):
    route_base = '/total_resource/api/'
    order_columns=["cpu","memory"]
    primary_key='pod'
    cols_width={
        "cluster": {"type": "ellip2", "width": 100},
        "project": {"type": "ellip2", "width": 100},
        "namespace": {"type": "ellip2", "width": 100},
        "node": {"type": "ellip2", "width": 150},
        "pod": {"type": "ellip2", "width": 500},
        "username": {"type": "ellip2", "width": 150},
        "cpu": {"type": "ellip2", "width": 100},
        "memory": {"type": "ellip2", "width": 100},
        "gpu": {"type": "ellip2", "width": 100}
    }
    label_columns={
        "cluster":"集群",
        "project":"项目组",
        "namespace":"空间",
        "pod":"容器",
        "username":"用户",
        "node": "节点",
        "cpu":"cpu使用",
        "memory":"内存使用",
        "gpu":"gpu使用"
    }
    ops_link = [
        {
            "text": "gpu资源监控",
            "url": conf.get('GRAFANA_GPU_PATH')
        },
        {
            "text": "集群负载",
            "url": conf.get('GRAFANA_CLUSTER_PATH')
        }
    ]
    label_title='整体资源'
    list_title = "运行中资源列表"
    page_size=1000
    enable_echart=True
    base_permissions=['can_list']
    list_columns = ['cluster','project','namespace','pod','username','node','cpu','memory','gpu']

    alert_config={
        conf.get('MODEL_URLS',{}).get('total_resource',''):node_traffic
    }

    def query_list(self, order_column,order_direction,page_index,page_size,filters=None,**kargs):

        lst=pod_resource()
        if order_column and lst:
            lst = sorted(lst, key=lambda d:float(d[order_column].split('/')[0])/float(d[order_column].split('/')[1]) if '/0' not in d[order_column] else 0, reverse = False if order_direction=='asc' else True)
        total_count=len(lst)
        return total_count,lst

    # @pysnooper.snoop()
    def echart_option(self,filters=None):
        global global_cluster_load

        if not global_cluster_load:
            node_resource_used['check_time']=None
            node_traffic()
        pod_resource_metric = pod_resource()
        # print(pod_resource_metric)
        # print(global_cluster_load)
        if not pod_resource_metric or not global_cluster_load:
            return {}

        # 获取不同集群当前的占用率
        data1=[]
        for cluster_name in global_cluster_load:
            traffic = 0
            num=0
            if int(global_cluster_load[cluster_name]['cpu_all']):
                traffic += int(global_cluster_load[cluster_name]['cpu_req'])/int(global_cluster_load[cluster_name]['cpu_all'])
                num+=1
            if int(global_cluster_load[cluster_name]['mem_all']):
                traffic += int(global_cluster_load[cluster_name]['mem_req'])/int(global_cluster_load[cluster_name]['mem_all'])
                num+=1
            if int(global_cluster_load[cluster_name]['gpu_all']):
                traffic += int(float(global_cluster_load[cluster_name]['gpu_req']))/int(float(global_cluster_load[cluster_name]['gpu_all']))
                num+=1
            if num:
                data1.append({
                    "name":cluster_name,
                    "value":traffic/num
                })


        # 获取不同人的资源占用总数
        metric={}
        for pod in pod_resource_metric:
            if pod['username'] not in metric:
                metric[pod['username']]=0
            # 1G 22，，1 核 44， 1 T4 1400
            metric[pod['username']] += int(pod['memory'].split("/")[1])
            metric[pod['username']] += int(pod['cpu'].split("/")[1])*2
            metric[pod['username']] += int(float(pod['gpu']))*60
        data2 = []
        for username in metric:
            if int(metric[username])>0:
                data2.append({
                    "name":username,
                    "value":int(metric[username])
                })

        # 获取不同人的资源利用率
        metric={}
        for pod in pod_resource_metric:
            if pod['username'] not in metric:
                metric[pod['username']]={
                    "cpu_req":0,
                    "cpu_all":0,
                    "mem_req":0,
                    "mem_all":0,
                    "gpu_req":0,
                    "gpu_all":0
                }

            metric[pod['username']]["mem_req"] += int(pod['memory'].split("/")[0])
            metric[pod['username']]["mem_all"] += int(pod['memory'].split("/")[1])
            metric[pod['username']]["cpu_req"] += int(pod['cpu'].split("/")[0])
            metric[pod['username']]["cpu_all"] += int(pod['cpu'].split("/")[1])
            metric[pod['username']]["gpu_req"] += int(float(pod['gpu'].split("/")[0]))

        data3 = []
        for username in metric:
            if metric[username]['mem_all']>0 and metric[username]['cpu_all']>0:
                value = metric[username]['mem_req']/metric[username]['mem_all'] / 2 + metric[username]['cpu_req']/metric[username]['cpu_all'] / 2
                data3.append({
                    "name":username,
                    "value":value
                })
        # print(data1)
        # print(data2)
        # print(data3)

        option = '''
        {
            "title": [
                {
                    "subtext": '集群资源占用率',
                    "left": '16.67%',
                    "top": '75%',
                    "textAlign": 'center'
                },
                {
                    "subtext": '用户资源占用',
                    "left": '50%',
                    "top": '75%',
                    "textAlign": 'center'
                },
                {
                    "subtext": '用户利用率',
                    "left": '83.33%',
                    "top": '75%',
                    "textAlign": 'center'
                }
            ],
            "tooltip": {
                "formatter": function (params) {
                        console.log(params)
                        if(params.seriesIndex==0){
                            return "集群占用率："+(params.value.toFixed(2)*100).toString()+"%"
                        }
                        
                        if(params.seriesIndex==1){
                            return "用户占用总量(内存)(按价格折算)："+params.value.toString()
                        }
                        if(params.seriesIndex==2){
                            return "用户利用率："+(params.value.toFixed(2)*100).toString()+"%"
                        }
                }
            },
            "series": [
                {
                    "type": 'pie',
                    "radius": '40%',
                    "center": ['50%', '50%'],
                    "data": {{data1}},
                    "label": {
                        "position": 'outer',
                        "alignTo": 'none',
                        "bleedMargin": 5
                    },
                    "left": 0,
                    "right": '66.6667%',
                    "top": 0,
                    "bottom": 0
                },
                {
                    "type": 'pie',
                    "radius": '40%',
                    "center": ['50%', '50%'],
                    "data": {{data2}},
                    "label": {
                        "position": 'outer',
                        "alignTo": 'labelLine',
                        "bleedMargin": 5
                    },
                    "left": '33.3333%',
                    "right": '33.3333%',
                    "top": 0,
                    "bottom": 0
                },
                {
                    "type": 'pie',
                    "radius": '40%',
                    "center": ['50%', '50%'],
                    "data": {{data3}},
                    "label": {
                        "position": 'outer',
                        "alignTo": 'edge',
                        "margin": 20
                    },
                    "left": '66.6667%',
                    "right": 0,
                    "top": 0,
                    "bottom": 0
                }
            ]
        }

        '''

        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(option)
        option = rtemplate.render(data1=data1,data2=data2,data3=data3)

        return option


appbuilder.add_api(Total_Resource_ModelView_Api)




