import copy
import math
from flask import Markup,g
from jinja2 import Environment, BaseLoader, DebugUndefined
from myapp import app, appbuilder, db
from flask import request
from .baseFormApi import (
    MyappFormRestApi
)
from flask_appbuilder.actions import action
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import datetime, time, json
import pysnooper
from myapp.utils.py.py_k8s import K8s

conf = app.config

pipeline_resource_used = {
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
    if not node_resource_used['check_time'] or node_resource_used['check_time'] < (datetime.datetime.now() - datetime.timedelta(seconds=10)):

        all_node_json = {}
        clusters = conf.get('CLUSTERS', {})
        for cluster_name in clusters:
            try:
                cluster = clusters[cluster_name]
                k8s_client = K8s(cluster.get('KUBECONFIG', ''))

                all_node = k8s_client.get_node()
                all_node_resource = k8s_client.get_all_node_allocated_resources()
                all_node_json[cluster_name] = {}
                for node in all_node:
                    all_node_json[cluster_name][node['hostip']] = node
                    node_allocated_resources = all_node_resource.get(node['name'], {
                        "used_cpu": 0,
                        "used_memory": 0,
                        "used_gpu": 0
                    })
                    # print(node_allocated_resources)
                    all_node_json[cluster_name][node['hostip']].update(node_allocated_resources)
            except Exception as e:
                print(e)

        node_resource_used['data'] = all_node_json
        node_resource_used['check_time'] = datetime.datetime.now()

    all_node_json = node_resource_used['data']
    # print(all_node_json)
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
        td_html % __("集群"), td_html % __("资源组"), td_html % __("机器"), td_html % __("机型"), td_html % __("cpu占用率"), td_html % __("内存占用率"),
        td_html % __("AI加速卡"))

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
            if nodes[ip]['labels'].get('gpu','')=='true':
                device = 'gpu/' + nodes[ip]['labels'].get('gpu-type', '')
            if nodes[ip]['labels'].get('vgpu', '') == 'true':
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
        grafana_url = "//" + cluster_config.get('HOST', request.host) + conf.get('GRAFANA_CLUSTER_PATH')
        for ip in nodes:
            node_dashboard_url = "//"+ cluster_config.get('HOST', request.host) + conf.get('K8S_DASHBOARD_CLUSTER') + '#/node/%s?namespace=default' % nodes[ip]['name']
            org = nodes[ip]['labels'].get('org', 'public')
            enable_train = nodes[ip]['labels'].get('train', 'true')
            ip_html = '<a target="_blank" href="%s">%s</a>' % (node_dashboard_url, ip if nodes[ip]['status']=='Ready' else f'<del>{ip}</del>')

            share = nodes[ip]['labels'].get('share', 'true')
            clolr = "#FFFFFF" if share == 'true' else '#F0F0F0'
            device = ''
            if nodes[ip]['labels'].get('cpu','') == 'true':
                device = 'cpu/'
            if nodes[ip]['labels'].get('gpu','')=='true':
                device = device+'gpu/'
            if nodes[ip]['labels'].get('vgpu','')=='true':
                device = device+'vgpu/'
            device = device + nodes[ip]['labels'].get('gpu-type', '')
            device=device.strip('/')

            gpu_used=0
            gpu_total=0
            gpu_mfrs='gpu'
            gpu_resource = conf.get('GPU_RESOURCE')
            for gpu_mfrs_temp in gpu_resource:
                gpu_used_temp = round(nodes[ip].get(f'used_{gpu_mfrs_temp}',0), 2)
                gpu_total_temp = nodes[ip][gpu_mfrs_temp]
                if gpu_total_temp>0.01:  # 存在指定卡型资源，就显示指定卡提供应简写
                    gpu_mfrs = gpu_mfrs_temp
                gpu_used += gpu_used_temp
                gpu_total += gpu_total_temp

            message += '<tr bgcolor="%s">%s %s %s %s %s %s %s<tr>' % (
                clolr,
                td_html % cluster_name,
                td_html % org,
                td_html % ip_html,
                td_html % ('<a target="blank" href="%s">%s</a>' % (grafana_url, device)),
                td_html % ("cpu:%s/%s" % (nodes[ip]['used_cpu'], nodes[ip]['cpu'])),
                td_html % ("mem:%s/%s" % (nodes[ip]['used_memory'], nodes[ip]['memory'])),
                # td_html % ("gpu:%s/%s" % (round(nodes[ip]['used_gpu'],2) if 'vgpu' in device else int(float(nodes[ip]['used_gpu'])), nodes[ip]['gpu'])),
                td_html % (f"{gpu_mfrs}:{gpu_used}/{gpu_total}"),

                # td_html % (','.join(list(set(nodes[ip]['user']))[0:1]))
            )

            global_cluster_load[cluster_name]['cpu_req'] += int(nodes[ip]['used_cpu'])
            global_cluster_load[cluster_name]['cpu_all'] += int(nodes[ip]['cpu'])
            global_cluster_load[cluster_name]['mem_req'] += int(nodes[ip]['used_memory'])
            global_cluster_load[cluster_name]['mem_all'] += int(nodes[ip]['memory'])
            global_cluster_load[cluster_name]['gpu_req'] += round(gpu_used, 2)
            global_cluster_load[cluster_name]['gpu_all'] += int(float(gpu_total))

    message = Markup('<table style="margin:20px">%s</table>' % message)

    data = {
        'content': message,
        'delay': 300000,
        'hit': True,
        'target': conf.get('MODEL_URLS', {}).get('total_resource', ''),
        'title': __('机器负载'),
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
    if not pipeline_resource_used['check_time'] or pipeline_resource_used['check_time'] < (datetime.datetime.now() - datetime.timedelta(seconds=10)):
        clusters = conf.get('CLUSTERS', {})
        all_tasks_json = {}
        for cluster_name in clusters:
            cluster = clusters[cluster_name]
            try:
                k8s_client = K8s(cluster.get('KUBECONFIG', ''))
                # 获取pod的资源占用
                all_tasks_json[cluster_name] = {}
                # print(all_node_json)
                for namespace in ['jupyter', 'pipeline', 'automl', 'service', 'aihub']:
                    all_tasks_json[cluster_name][namespace] = {}
                    all_pods = k8s_client.get_pods(namespace=namespace)
                    for pod in all_pods:
                        org = pod['node_selector'].get("org", 'public')
                        if org not in all_tasks_json[cluster_name][namespace]:
                            all_tasks_json[cluster_name][namespace][org] = {}
                        if pod['status'] == 'Running' or pod['status_more'].get('phase','')=='Running':
                            user = pod['labels'].get('user', pod['labels'].get('username', pod['labels'].get('run-rtx',pod['labels'].get('rtx-user','admin'))))
                            if user:
                                request_gpu = 0
                                # 所有ai卡都算入xpu卡中
                                for gpu_mfrs in list(conf.get('GPU_RESOURCE',{}).keys()):
                                    request_gpu +=float(pod.get(gpu_mfrs,0))

                                all_tasks_json[cluster_name][namespace][org][pod['name']] = {}
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['username'] = user
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['host_ip'] = pod['host_ip']
                                # print(namespace,pod)
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['request_memory'] = pod['memory']
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['request_cpu'] = pod['cpu']
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['request_gpu'] = request_gpu
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_memory'] = '0'
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_cpu'] = '0'
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['used_gpu'] = '0'
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['label'] = pod['labels']
                                all_tasks_json[cluster_name][namespace][org][pod['name']]['annotations'] = pod['annotations']
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
    all_pod_resource = []
    for cluster_name in all_tasks_json:
        cluster_config = conf.get('CLUSTERS', {}).get(cluster_name, {})
        for namespace in all_tasks_json[cluster_name]:
            for org in all_tasks_json[cluster_name][namespace]:
                for pod_name in all_tasks_json[cluster_name][namespace][org]:
                    pod = all_tasks_json[cluster_name][namespace][org][pod_name]
                    dashboard_url = f'/k8s/web/search/{cluster_name}/{namespace}/{pod_name}'
                    task_grafana_url = "//" + cluster_config.get('HOST', request.host) + conf.get('GRAFANA_TASK_PATH')
                    node_grafana_url = "//" + cluster_config.get('HOST', request.host) + conf.get('GRAFANA_NODE_PATH')
                    pod_resource={
                        "cluster":cluster_name,
                        'project':pod['annotations'].get('project','unknown'),
                        "resource_group":org,
                        "namespace":Markup('<a target="blank" href="%s">%s</a>' % (dashboard_url, namespace)),
                        "pod":Markup('<a target="blank" href="%s">%s</a>' % (task_grafana_url + pod_name, pod_name)),
                        "pod_info":f"{cluster_name}:{namespace}:{org}:{pod_name}",
                        "label":pod['label'],
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
    order_columns = ["cpu", "memory"]
    primary_key = 'pod_info'
    cols_width = {
        "cluster": {"type": "ellip2", "width": 100},
        "resource_group": {"type": "ellip2", "width": 100},
        "project": {"type": "ellip2", "width": 100},
        "namespace": {"type": "ellip2", "width": 100},
        "node": {"type": "ellip2", "width": 150},
        "pod": {"type": "ellip2", "width": 400},
        "username": {"type": "ellip2", "width": 100},
        "cpu": {"type": "ellip2", "width": 100},
        "memory": {"type": "ellip2", "width": 100},
        "gpu": {"type": "ellip2", "width": 100}
    }
    label_columns = {
        "cluster": _("集群"),
        "project": _("项目组"),
        "resource_group": _("资源组"),
        "namespace": _("类型"),
        "label": _("标签"),
        "pod": _("容器(资源使用)"),
        "username": _("用户"),
        "node": _("节点(资源使用)"),
        "cpu": _("cpu使用"),
        "memory": _("内存使用"),
        "gpu": _("AI卡使用")
    }
    ops_link = [
        {
            "text": _("gpu资源监控"),
            "url": conf.get('GRAFANA_GPU_PATH')
        },
        {
            "text": _("集群负载"),
            "url": conf.get('GRAFANA_CLUSTER_PATH')
        }
    ]
    label_title = _('整体资源')
    list_title = _("运行中资源列表")
    page_size = 1000
    enable_echart = True
    base_permissions = ['can_list']
    list_columns = ['cluster', 'resource_group', 'namespace', 'pod', 'username', 'node', 'cpu', 'memory', 'gpu']

    alert_config = {
        conf.get('MODEL_URLS', {}).get('total_resource', ''): node_traffic
    }

    def query_list(self, order_column, order_direction, page_index, page_size, filters=None, **kargs):

        lst = pod_resource()

        # 非管理员只查看自己的
        if not g.user.is_admin():
            lst = [pod for pod in lst if pod['username']==g.user.username]

        if order_column and lst:
            lst = sorted(lst, key=lambda d:float(d[order_column].split('/')[0])/float(d[order_column].split('/')[1]) if '/0' not in d[order_column] else 0, reverse = False if order_direction=='asc' else True)
        total_count=len(lst)
        return total_count,lst

    # @pysnooper.snoop()
    def echart_option(self, filters=None):
        global global_cluster_load

        if not global_cluster_load:
            node_resource_used['check_time'] = None
            node_traffic()

        from myapp.utils.py.py_prometheus import Prometheus
        prometheus = Prometheus(conf.get('PROMETHEUS', 'prometheus-k8s.monitoring:9090'))
        # prometheus = Prometheus('9.135.92.226:15046')

        pod_resource_metric = prometheus.get_resource_metric()

        all_resource={
            "mem_all": sum([int(global_cluster_load[cluster_name]['mem_all']) for cluster_name in global_cluster_load]),
            "cpu_all": sum([int(global_cluster_load[cluster_name]['cpu_all']) for cluster_name in global_cluster_load]),
            "gpu_all": sum([int(global_cluster_load[cluster_name]['gpu_all']) for cluster_name in global_cluster_load]),
        }
        all_resource_req = {
            "mem_req": sum([int(global_cluster_load[cluster_name]['mem_req']) for cluster_name in global_cluster_load]),
            "cpu_req": sum([int(global_cluster_load[cluster_name]['cpu_req']) for cluster_name in global_cluster_load]),
            "gpu_req": sum([int(global_cluster_load[cluster_name]['gpu_req']) for cluster_name in global_cluster_load]),
        }
        all_resource_used = {
            "mem_used": sum([pod_resource_metric[x].get('memory',0) for x in pod_resource_metric]),
            "cpu_used": sum([pod_resource_metric[x].get('cpu',0) for x in pod_resource_metric]),
            "gpu_used": sum([pod_resource_metric[x].get('gpu',0) for x in pod_resource_metric]),
        }

        resource_options = open('myapp/utils/echart/resource.txt').read()
        chat1 = copy.deepcopy(resource_options)
        chat1 = chat1.replace('MEM_NAME', __('内存总量(G)')).replace('MEM_CENTER_X', '7%').replace('MEM_VALUE', str(int(all_resource['mem_all'])))
        chat1 = chat1.replace('CPU_NAME', __('CPU总量(核)')).replace('CPU_CENTER_X', '17%').replace('CPU_VALUE', str(int(all_resource['cpu_all'])))
        chat1 = chat1.replace('GPU_NAME', __('GPU总量(卡)')).replace('GPU_CENTER_X', '27%').replace('GPU_VALUE', str(int(all_resource['gpu_all'])))
        chat1 = chat1.replace('MEM_MAX', str(int(all_resource['mem_all']*2))).replace('CPU_MAX', str(int(all_resource['cpu_all']*2))).replace('GPU_MAX', str(int(all_resource['gpu_all']*2)))
        chat1 = chat1.strip('\n').strip(' ').strip(',')

        chat2 = copy.deepcopy(resource_options)
        chat2 = chat2.replace('MEM_NAME',__('内存占用率')).replace('MEM_CENTER_X','40%').replace('MEM_VALUE',str(int(100*all_resource_req['mem_req']/(0.001+all_resource['mem_all']))))
        chat2 = chat2.replace('CPU_NAME',__('CPU占用率')).replace('CPU_CENTER_X','50%').replace('CPU_VALUE',str(int(100*all_resource_req['cpu_req']/(0.001+all_resource['cpu_all']))))
        chat2 = chat2.replace('GPU_NAME',__('GPU占用率')).replace('GPU_CENTER_X','60%').replace('GPU_VALUE',str(int(100*all_resource_req['gpu_req']/(0.001+all_resource['gpu_all']))))
        chat2 = chat2.replace('MEM_MAX', '100').replace('CPU_MAX', '100').replace('GPU_MAX', '100').replace('{a|{value}}','{a|{value}%}')
        chat2 = chat2.strip('\n').strip(' ').strip(',')

        chat3 = copy.deepcopy(resource_options)
        chat3 = chat3.replace('MEM_NAME',__('内存利用率')).replace('MEM_CENTER_X','73%').replace('MEM_VALUE',str(min(100,int(100*all_resource_used['mem_used']/(0.001+all_resource['mem_all'])))))
        chat3 = chat3.replace('CPU_NAME',__('CPU利用率')).replace('CPU_CENTER_X','83%').replace('CPU_VALUE',str(min(100,int(100*all_resource_used['cpu_used']/(0.001+all_resource['cpu_all'])))))
        chat3 = chat3.replace('GPU_NAME',__('GPU利用率')).replace('GPU_CENTER_X','93%').replace('GPU_VALUE',str(min(100,int(100*all_resource_used['gpu_used']/(0.001+all_resource['gpu_all'])))))
        chat3 = chat3.replace('MEM_MAX', '100').replace('CPU_MAX', '100').replace('GPU_MAX', '100').replace('{a|{value}}','{a|{value}%}')
        chat3 = chat3.strip('\n').strip(' ').strip(',')
        DATA = ',\n'.join([chat1,chat2,chat3])
        option = '''
        {
            "title": [
                {
                    "subtext": '集群信息',
                    "left": '17%',
                    "top": '75%',
                    "textAlign": 'center'
                },
                {
                    "subtext": '资源占用率',
                    "left": '50%',
                    "top": '75%',
                    "textAlign": 'center'
                },
                {
                    "subtext": '资源利用率',
                    "left": '83%',
                    "top": '75%',
                    "textAlign": 'center'
                }
            ],
            "series": [
                DATA
            ]
        }

        '''
        option=option.replace('DATA',DATA)
        option = option.replace('集群信息',__('集群信息')).replace('资源占用率',__('资源占用率')).replace('资源利用率',__('资源利用率'))

        return option

    message = '如果是pipeline命名空间，按照run-id进行删除。\n如果是jupyter命名空间，直接删除pod\n'
    @action("muldelete", "清理", "清理选中的pipeline，service，notebook的pod，但并不删除他们?", "fa-trash", single=False)
    # @pysnooper.snoop()
    def muldelete(self, items):
        pod_resource()
        all_task_resource = pipeline_resource_used['data']
        clusters = conf.get('CLUSTERS', {})
        for item in items:
            try:
                cluster_name,namespace,org,pod_name = item.split(":")
                cluster = clusters[cluster_name]
                print(cluster_name,namespace,pod_name)
                k8s_client = K8s(cluster.get('KUBECONFIG', ''))
                pod = all_task_resource[cluster_name][namespace][org][pod_name]
                # 如果是jupyter命名空间，直接删除pod
                if namespace=='jupyter':
                    k8s_client.delete_pods(namespace=namespace,pod_name=pod_name)
                    k8s_client.delete_service(namespace=namespace,name=pod_name)
                    service_external_name = (pod_name + "-external").lower()[:60].strip('-')
                    k8s_client.delete_service(namespace=namespace, name=service_external_name)
                    k8s_client.delete_istio_ingress(namespace=namespace, name=pod_name)
                # 如果是service命名空间，需要删除deployment和service和虚拟服务
                if namespace=='service':
                    service_name = pod['label'].get("app",'')
                    if service_name:
                        service_external_name = (service_name + "-external").lower()[:60].strip('-')
                        k8s_client.delete_deployment(namespace=namespace, name=service_name)
                        k8s_client.delete_service(namespace=namespace, name=service_name)
                        k8s_client.delete_service(namespace=namespace, name=service_external_name)
                        k8s_client.delete_istio_ingress(namespace=namespace, name=service_name)
                # 如果是aihub命名空间，需要删除 deployemnt和service和虚拟服务
                if namespace == 'aihub':
                    service_name = pod['label'].get("app", '')
                    if service_name:
                        k8s_client.delete_deployment(namespace=namespace, name=service_name)
                        k8s_client.delete_service(namespace=namespace, name=service_name)
                        k8s_client.delete_istio_ingress(namespace=namespace, name=service_name)
                # 如果是pipeline命名空间，按照run-id进行删除，这里先只删除pod，也会造成任务停止
                if namespace == 'pipeline':
                    k8s_client.delete_pods(namespace=namespace, pod_name=pod_name)

                if namespace == 'automl':
                    vcjob_name = pod['label'].get("app", '')
                    k8s_client.delete_pods(namespace=namespace, pod_name=pod_name)
                    if vcjob_name:
                        k8s_client.delete_volcano(namespace=namespace, name=vcjob_name)
                        k8s_client.delete_service(namespace=namespace,labels={'app':vcjob_name})
                        k8s_client.delete_istio_ingress(namespace=namespace,name=pod_name)
            except Exception as e:
                print(e)

        return json.dumps(
            {
                "message": 'delete: '+str(items),
            }, indent=4, ensure_ascii=False
        )

    # # @pysnooper.snoop()
    # def add_more_info(self, response, **kwargs):
    #     if not g.user or not g.user.is_admin():
    #         response['action'] = {}

appbuilder.add_api(Total_Resource_ModelView_Api)



