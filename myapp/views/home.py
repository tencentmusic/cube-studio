
from flask import (
    current_app,
    abort,
    flash,
    g,
    Markup,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)
from myapp.utils.py.py_k8s import K8s
import datetime,json
from flask import Flask, jsonify
import pysnooper
from apispec import yaml_utils
from flask import Blueprint, current_app, jsonify, make_response, request
from flask_babel import lazy_gettext as _
from myapp import conf, db, get_feature_flags, security_manager,event_logger
import yaml
from myapp.views.base import BaseMyappView

from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from myapp import app, appbuilder

resource_used = {
    "check_time": None,
    "data": {}
}

class Myapp(BaseMyappView):
    route_base='/myapp'
    default_view = 'welcome'   # 设置进入蓝图的默认访问视图（没有设置网址的情况下）

    @expose('/welcome')
    @expose('/profile/<username>/')
    def welcome(self,username=None):
        if not g.user or not g.user.get_id():
            return redirect(appbuilder.get_url_for_login)
        if username:
            msg = 'Hello ' + username + " !"
        else:
            msg = 'Hello '+g.user.username+" !"

        # 返回模板
        return self.render_template('hello.html', msg=msg)

    @expose('/home')
    def home(self):
        # 返回模板
        return self.render_template('home.html')

    @expose("/web/log/<cluster_name>/<namespace>/<pod_name>", methods=["GET",])
    def web_log(self,cluster_name,namespace,pod_name):
        from myapp.utils.py.py_k8s import K8s
        all_clusters = conf.get('CLUSTERS',{})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name]['KUBECONFIG']
            pod_url = all_clusters[cluster_name].get('K8S_DASHBOARD_CLUSTER') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, pod_name)
        else:
            kubeconfig = None
            pod_url = conf.get('K8S_DASHBOARD_CLUSTER') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, pod_name)

        k8s = K8s(kubeconfig)
        pod = k8s.get_pods(namespace=namespace, pod_name=pod_name)
        if pod:
            pod = pod[0]
            flash('当前pod状态：%s'%pod['status'],category='warning')
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',     #  kd-logs-container  :nth-of-type(0)
            "delay": 2000,
            "loading":True,
            "currentHeight": 128
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)


    @expose("/web/debug/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET", "POST"])
    def web_debug(self,cluster_name,namespace,pod_name,container_name):
        cluster=conf.get('CLUSTERS',{})
        if cluster_name in cluster:
            pod_url = cluster[cluster_name].get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name,container_name, namespace)
        else:
            pod_url = conf.get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name, container_name, namespace)
        print(pod_url)
        data = {
            "url": pod_url,
            "target":'div.kd-scroll-container', #  'div.kd-scroll-container.ng-star-inserted',
            "delay": 2000,
            "loading": True
        }
        # 返回模板
        if cluster_name==conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)


    @expose('/schedule/node/<ip>')
    def schedule_node(self,ip):
        all_node_json = resource_used['data']
        for cluster_name in all_node_json:
            nodes = all_node_json[cluster_name]
            if ip in nodes:
                clusters = conf.get('CLUSTERS', {})
                cluster = clusters[cluster_name]
                k8s_client = K8s(cluster['KUBECONFIG'])
                # 获取最新的节点信息
                nodes = k8s_client.get_node(ip=ip)
                if nodes:
                    node = nodes[0]
                    enable_train = node['labels'].get('train','true')
                    k8s_client.label_node([ip],{"train":"false" if enable_train=='true' else "true"})
                    break

        return redirect('/myapp/home')


    # from myapp import tracer

    @expose('/feature/check')
    # @trace(tracer,depth=1,trace_content='line')
    def featureCheck(self):
        url = request.values.get("url", type=str, default=None)
        if '/myapp/home' in url:
            if 1 or not resource_used['check_time'] or resource_used['check_time']<(datetime.datetime.now()-datetime.timedelta(minutes=10)):
                clusters = conf.get('CLUSTERS', {})
                for cluster_name in clusters:
                    cluster = clusters[cluster_name]
                    k8s_client = K8s(cluster['KUBECONFIG'])

                    all_node = k8s_client.get_node()
                    all_node_json = {}
                    for node in all_node:   # list 转dict
                        ip = node['hostip']
                        if 'cpu' in node['labels'] or 'gpu' in node['labels']:
                            all_node_json[ip]=node
                            all_node_json[ip]['used_memory'] = []
                            all_node_json[ip]['used_cpu'] = []
                            all_node_json[ip]['used_gpu'] = []
                            all_node_json[ip]['user'] = []

                    # print(all_node_json)
                    for namespace in ['jupyter', 'pipeline', 'katib', 'service']:
                        all_pods = k8s_client.get_pods(namespace=namespace)
                        for pod in all_pods:
                            if pod['status'] == 'Running' and pod['host_ip'] in all_node_json:
                                # print(namespace,pod)
                                all_node_json[pod['host_ip']]['used_memory'].append(pod['memory'])
                                all_node_json[pod['host_ip']]['used_cpu'].append(pod['cpu'])
                                all_node_json[pod['host_ip']]['used_gpu'].append(pod['gpu'])

                                # user = pod['labels'].get('user','')
                                # if not user:
                                #     user = pod['labels'].get('run-rtx','')
                                # if not user:
                                #     user = pod['labels'].get('rtx-user','')
                                # if user:
                                #     all_node_json[pod['host_ip']]['user'].append(user)
                                # print(all_node_json[pod['host_ip']])

                    for node in all_node_json:
                        all_node_json[node]['used_memory'] = int(sum(all_node_json[node]['used_memory']))
                        all_node_json[node]['used_cpu'] = int(sum(all_node_json[node]['used_cpu']))
                        all_node_json[node]['used_gpu'] = int(sum(all_node_json[node]['used_gpu']))

                    resource_used['data'][cluster_name]=all_node_json
                resource_used['check_time']=datetime.datetime.now()

            all_node_json = resource_used['data']

            # 数据格式说明 dict:
            # 'delay': Integer 延时隐藏 单位: 毫秒 0为不隐藏
            # 'hit': Boolean 是否命中
            # 'target': String 当前目标
            # 'type': String 类型 目前仅支持html类型
            # 'title': String 标题
            # 'content': String 内容html内容
            # /static/appbuilder/mnt/make_pipeline.mp4
            message = ''
            td_html='<td style="border: 1px solid black;padding: 10px">%s</th>'
            message += "<tr>%s %s %s %s %s %s %s<tr>" %(td_html%"集群",td_html%"资源组(监控)", td_html%"机器(进出)", td_html%"机型", td_html%"cpu占用率", td_html%"内存占用率", td_html%"gpu占用率")
            global_cluster_load={}
            for cluster_name in all_node_json:
                global_cluster_load[cluster_name]={
                    "cpu_req":0,
                    "cpu_all":0,
                    "mem_req":0,
                    "mem_all":0,
                    "gpu_req":0,
                    "gpu_all":0
                }
                nodes = all_node_json[cluster_name]
                # nodes = sorted(nodes.items(), key=lambda item: item[1]['labels'].get('org','public'))
                # ips = [node[0] for node in nodes]
                # values = [node[1] for node in nodes]
                # nodes = dict(zip(ips,values))

                # 按项目组和设备类型分组
                stored_nodes={}
                for ip in nodes:
                    org = nodes[ip]['labels'].get('org','public')
                    device = 'gpu/'+nodes[ip]['labels'].get('gpu-type','') if 'gpu' in nodes[ip]['labels'] else 'cpu'
                    if org not in stored_nodes:
                        stored_nodes[org]={}
                    if device not in stored_nodes[org]:
                        stored_nodes[org][device]={}
                    stored_nodes[org][device][ip]=nodes[ip]
                nodes={}
                for org in stored_nodes:
                    for device in stored_nodes[org]:
                        nodes.update(stored_nodes[org][device])

                cluster_config=conf.get('CLUSTERS',{}).get(cluster_name,{})
                grafana_url = cluster_config.get('GRAFANA_HOST','').strip('/')+conf.get('GRAFANA_CLUSTER_PATH')
                for ip in nodes:
                    org = nodes[ip]['labels'].get('org','public')
                    enable_train=nodes[ip]['labels'].get('train','true')
                    if g.user.is_admin():
                        if enable_train=='true':
                            ip_html='<a href="%s">%s</a>'%("/myapp/schedule/node/%s"%ip,ip)
                        else:
                            ip_html = '<a href="%s"><strike>%s</strike></a>' % ("/myapp/schedule/node/%s" % ip, ip)
                    else:
                        if enable_train=='true':
                            ip_html=ip
                        else:
                            ip_html = '<strike>%s</strike>' % (ip,)
                    share = nodes[ip]['labels'].get('share','true')
                    clolr = "#FFFFFF" if share=='true' else '#F0F0F0'
                    message+='<tr bgcolor="%s">%s %s %s %s %s %s %s<tr>'%(
                        clolr,
                        td_html%cluster_name,
                        td_html % ('<a target="blank" href="%s">%s</a>'%(grafana_url+org,org)),
                        td_html%ip_html,
                        td_html%('gpu/'+nodes[ip]['labels'].get('gpu-type','') if 'gpu' in nodes[ip]['labels'] else 'cpu'),
                        td_html%("cpu:%s/%s"%(nodes[ip]['used_cpu'],nodes[ip]['cpu'])),
                        td_html%("mem:%s/%s"%(nodes[ip]['used_memory'],nodes[ip]['memory'])),
                        td_html%("gpu:%s/%s"%(nodes[ip]['used_gpu'],nodes[ip]['gpu'])),
                        # td_html % (','.join(list(set(nodes[ip]['user']))[0:1]))
                    )

                    global_cluster_load[cluster_name]['cpu_req']+=int(nodes[ip]['used_cpu'])
                    global_cluster_load[cluster_name]['cpu_all'] += int(nodes[ip]['cpu'])
                    global_cluster_load[cluster_name]['mem_req']+=int(nodes[ip]['used_memory'])
                    global_cluster_load[cluster_name]['mem_all'] += int(nodes[ip]['memory'])
                    global_cluster_load[cluster_name]['gpu_req']+=int(nodes[ip]['used_gpu'])
                    global_cluster_load[cluster_name]['gpu_all'] += int(nodes[ip]['gpu'])



            message=Markup(f'<table>%s</table>'%message)
            # print(message)
            cluster_global_info=''
            # for cluster_name in global_cluster_load:
            #     cluster_global_info+='\n集群:%s,CPU:%s/%s,MEM:%s/%s,GPU::%s/%s'%(
            #         cluster_name,
            #         global_cluster_load[cluster_name]['cpu_req'],global_cluster_load[cluster_name]['cpu_all'],
            #         global_cluster_load[cluster_name]['mem_req'], global_cluster_load[cluster_name]['mem_all'],
            #         global_cluster_load[cluster_name]['gpu_req'], global_cluster_load[cluster_name]['gpu_all'],
            #     )

            data = {
                'content': message,
                'delay': 300000,
                'hit': True,
                'target': url,
                'title': '当前负载(%s)'%cluster_global_info,
                'type': 'html',
            }
            # 返回模板
            return jsonify(data)
        return jsonify({})


# add_view_no_menu添加视图，但是没有菜单栏显示
appbuilder.add_view_no_menu(Myapp)

