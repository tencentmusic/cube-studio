import copy
import json
import re
import humanize
import flask
import pysnooper, os
from flask_appbuilder.baseviews import expose_api
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from dateutil.tz import tzutc
from myapp import app, conf
from myapp.utils.py.py_k8s import K8s, K8SStreamThread
from flask import g, flash, request, render_template, send_from_directory, send_file, make_response, Markup, jsonify, redirect
import datetime, time
from myapp import app, appbuilder, db, event_logger,cache
from .base import BaseMyappView
from flask_appbuilder import CompactCRUDMixin, expose

from myapp.utils.py.py_k8s import K8s
default_status_icon = '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#f3b146" p-id="7364"></path></svg>'

status_icon = {
    "Running": '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#33c43c" p-id="7364"></path></svg>',
    "Error": '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#d81e06" p-id="7364"></path></svg>',
    "Failed": '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#d81e06" p-id="7364"></path></svg>',
    "CrashLoopBackOff": '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#d81e06" p-id="7364"></path></svg>',
    'Succeeded': '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#155a1a" p-id="7364"></path></svg>',
    'Completed': '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#155a1a" p-id="7364"></path></svg>',
    'Terminating': '<svg t="1755008266851" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7363" id="mx_n_1755008266852" width="15" height="15"><path d="M512 512m-370.78857422 0a370.78857422 370.78857422 0 1 0 741.57714844 0 370.78857422 370.78857422 0 1 0-741.57714844 0Z" fill="#757575" p-id="7364"></path></svg>'
}
class K8s_View(BaseMyappView):
    route_base = '/k8s'

    # 打开pod日志界面
    @expose_api(description="打开pod日志界面",url="/watch/log/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET", ])
    def watch_log(self, cluster_name, namespace, pod_name, container_name):

        all_clusters = conf.get('CLUSTERS', {})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
        else:
            kubeconfig = None

        k8s_client = K8s(kubeconfig)
        pods = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        if pods:
            pod = pods[0]
            if pod['username']!=g.user.username and not g.user.is_admin():
                return {
                    "message": _('您暂无权限查看此pod日志，进管理员和创建者可以查看'),
                }

        data = {
            "url": '/k8s/stream/log',
            "server_event_name": "server_event_name",
            "user_event_name": cluster_name + "_" + namespace + "_" + pod_name + "_" + container_name,
            "cluster_name": cluster_name,
            "namespace_name": namespace,
            "pod_name": pod_name,
            "container_name": container_name,
        }
        # print(data)
        return self.render_template('log.html', data=data)

    # from myapp import socketio
    # # 实时获取pod日志
    # @socketio.on("k8s_stream_log",namespace='/k8s/stream/log')
    # # @pysnooper.snoop()
    # def stream_log(*args,**kwargs):
    #     import flask_socketio
    #     print(args)
    #     print(kwargs)
    #     message = args[0]
    #     message = json.loads(message) if type(message)==str else message
    #     cluster_name = message.get('cluster_name','')
    #     namespace_name = message.get('namespace_name','')
    #     pod_name = message.get('pod_name','')
    #     container_name = message.get('container_name','')
    #     user_event_name = message.get('user_event_name','')
    #     try:
    #         all_clusters = conf.get('CLUSTERS',{})
    #         if cluster_name in all_clusters:
    #             kubeconfig = all_clusters[cluster_name].get('KUBECONFIG','')
    #         else:
    #             kubeconfig = None
    #
    #         k8s = K8s(kubeconfig)
    #         stream = k8s.get_pod_log_stream(namespace=namespace_name, name=pod_name,container=container_name)
    #         if stream:
    #             for s in stream:
    #                 if s:
    #                     message = Markup(s.decode('utf-8'))
    #                     print(message)
    #                     flask_socketio.emit(user_event_name,message)
    #                 else:
    #                     break
    #     except Exception as e:
    #         print(e)
    #         flask_socketio.emit(user_event_name, str(e))

    # 打开pod执行命令界面
    @expose_api(description="打开pod执行命令界面",url="/watch/exec/<cluster_name>/<namespace>/<pod_name>/<container_name>")
    def watch_exec(self, cluster_name, namespace, pod_name, container_name):

        all_clusters = conf.get('CLUSTERS', {})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
        else:
            kubeconfig = None

        k8s_client = K8s(kubeconfig)
        pods = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        if pods:
            pod = pods[0]
            if pod['username']!=g.user.username and not g.user.is_admin():
                return {
                    "message": _('您暂无权限查看此pod日志，进管理员和创建者可以查看'),
                }

        data = {
            "ws_url": f"/k8s/stream/exec/{cluster_name}/{namespace}/{pod_name}/{container_name}"
        }
        # 返回模板
        return self.render_template('terminal.html', data=data)

    #
    # # 实时获取pod执行结果
    # from flask_sockets import Sockets
    # sockets = Sockets(app)
    # @sockets.route("/k8s/stream/exec/<cluster_name>/<namespace>/<pod_name>/<container_name>/<cols>/<rows>")
    # def stream_exec(ws,cluster_name,namespace,pod_name,container_name,cols,rows):
    #
    #     all_clusters = conf.get('CLUSTERS',{})
    #     if cluster_name in all_clusters:
    #         kubeconfig = all_clusters[cluster_name].get('KUBECONFIG','')
    #     else:
    #         kubeconfig = None
    #
    #     k8s = K8s(kubeconfig)
    #
    #     try:
    #         container_stream = k8s.terminal_start(namespace, pod_name, container_name,cols,rows)
    #     except Exception as err:
    #         print('Connect container error: {}'.format(err))
    #         ws.close()
    #         return
    #
    #     kub_stream = K8SStreamThread(ws, container_stream)
    #     kub_stream.start()
    #
    #     print('Start terminal')
    #     try:
    #         while not ws.closed:
    #             message = ws.receive()
    #             if message is not None:
    #                 if message != '__ping__':
    #                     container_stream.write_stdin(message)
    #         container_stream.write_stdin('exit\r')
    #     except Exception as err:
    #         print('Connect container error: {}'.format(err))
    #     finally:
    #         container_stream.close()
    #         ws.close()

    # 下载获取pod日志
    @expose_api(description="下载获取pod日志",url="/download/log/<cluster_name>/<namespace>/<pod_name>")
    def download_log(self, cluster_name, namespace, pod_name):
        try:
            all_clusters = conf.get('CLUSTERS', {})
            if cluster_name in all_clusters:
                kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
            else:
                kubeconfig = None

            k8s = K8s(kubeconfig)
            logs = k8s.download_pod_log(namespace=namespace, name=pod_name)
            file = open(pod_name, mode='w')
            file.write(logs)
            file.close()
            response = make_response(send_file(pod_name, as_attachment=True, conditional=True))
            return response
        except Exception as e:
            print(e)
            return str(e)

    # 返回获取pod日志
    @expose_api(description="返回获取pod日志",url="/read/log/<cluster_name>/<namespace>/<pod_name>")
    @expose_api(description="返回获取pod日志",url="/read/log/<cluster_name>/<namespace>/<pod_name>/<container>")
    @expose_api(description="返回获取pod日志",url="/read/log/<cluster_name>/<namespace>/<pod_name>/<container>/<tail>")
    def read_log(self, cluster_name, namespace, pod_name, container=None, tail=None):
        try:
            all_clusters = conf.get('CLUSTERS', {})
            if cluster_name in all_clusters:
                kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
            else:
                kubeconfig = None

            k8s = K8s(kubeconfig)
            if not tail:
                logs = k8s.download_pod_log(namespace=namespace, name=pod_name, container=container)
            elif 's' in tail:
                logs = k8s.download_pod_log(namespace=namespace, name=pod_name,container=container,since_seconds=tail.replace('s', ''))
            elif int(tail)>100000000:
                logs = k8s.download_pod_log(namespace=namespace, name=pod_name, container=container,since_time=tail)
            else:
                logs = k8s.download_pod_log(namespace=namespace, name=pod_name, container=container, tail_lines=tail)

            import re
            # 删除 ANSI 转义序列
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            logs = ansi_escape.sub('', logs)
            return logs
        except Exception as e:
            print(e)
            return str(e)

    # 返回获取pod的信息
    @expose_api(description="返回获取pod的信息",url="/read/pod/<cluster_name>/<namespace>/<pod_name>")
    def read_pod(self, cluster_name, namespace, pod_name):
        try:
            all_clusters = conf.get('CLUSTERS', {})
            if cluster_name in all_clusters:
                kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
            else:
                kubeconfig = None

            k8s_client = K8s(kubeconfig)
            pods = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
            if pods:
                pod = pods[0]
                pod['events'] = k8s_client.get_pod_event(namespace=namespace, pod_name=pod_name)
                return jsonify({
                    "status": 0,
                    "message": "",
                    "result": pod
                })
            else:
                return jsonify({
                    "status": 1,
                    "message": __("pod不存在"),
                    "result": {}
                })

        except Exception as e:
            print(e)
            response = make_response(str(e))
            response.status_code = 500
            return response

    # 返回获取terminating不掉的pod的信息
    @expose_api(description="返回获取terminating不掉的pod的信息",url="/read/pod/terminating")
    @expose_api(description="返回获取terminating不掉的pod的信息",url="/read/pod/terminating/<namespace>")
    # @pysnooper.snoop()
    def read_terminating_pod(self, namespace='service'):
        try:
            terminating_pods = cache.get('terminating_pods')
            if not terminating_pods:
                terminating_pods={}

                clusters = conf.get('CLUSTERS', {})
                for cluster_name in clusters:
                    try:
                        terminating_pods[cluster_name] = {}  # 重置，重新查询
                        cluster = clusters[cluster_name]
                        k8s_client = K8s(cluster.get('KUBECONFIG', ''))

                        events = [item.to_dict() for item in k8s_client.v1.list_namespaced_event(namespace=namespace,field_selector="type=Warning").items]  # ,field_selector=f'source.host={ip}'
                        # pod_names = [pod.metadata.name for pod in k8s_client.v1.list_namespaced_pod(namespace='service').items]
                        pods = k8s_client.get_pods(namespace=namespace)
                        pods_dict = {}
                        for pod in pods:
                            pods_dict[pod['name']] = pod
                        pod_names = [pod['name'] for pod in pods]

                        for event in events:
                            event['time'] = (event['first_timestamp'] + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') if event.get('first_timestamp', None) else None
                            if not event['time']:
                                event['time'] = (event['event_time'] + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') if event.get('event_time', None) else None
                            host = event.get("source", {}).get("host", '')
                            if event['type'] == 'Warning' and event['reason'] == 'FailedKillPod' and event['time'] < (datetime.datetime.now() - datetime.timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'):
                                # print(json.dumps(event,indent=4, ensure_ascii=False, default=str))
                                pod_name = event.get('involved_object', {}).get('name', '')
                                if pod_name in pod_names:
                                    terminating_pods[cluster_name][pod_name] = {
                                        "namespace": namespace,
                                        "host": host,
                                        "begin": event['time'],
                                        "username": pods_dict.get(pod_name, {}).get("username", ''),
                                        "label": pods_dict.get(pod_name, {}).get("label", '')
                                    }
                    except Exception as e:
                        print(e)

                cache.set('terminating_pods', terminating_pods,timeout=300)

            return jsonify(terminating_pods)

        except Exception as e:
            print(e)
            response = make_response(str(e))
            response.status_code = 500
            return response

    # 强制删除pod的信息
    @expose_api(description=" 强制删除pod的信息",url="/web/delete/pod/<cluster_name>/<namespace>/<pod_name>")
    @expose_api(description=" 强制删除pod的信息",url="/delete/pod/<cluster_name>/<namespace>/<pod_name>")
    def delete_pod(self, cluster_name, namespace, pod_name):
        try:
            cluster = conf.get('CLUSTERS', {}).get(cluster_name,{})
            k8s_client = K8s(cluster.get('KUBECONFIG', ''))
            pods = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
            if pods:
                pod = pods[0]
                if pod['username'] != g.user.username and not g.user.is_admin():
                    return {
                        "message": _('您暂无权限删除此pod，仅管理员和创建者可以查看'),
                    }
                k8s_client.v1.delete_namespaced_pod(pod['name'], namespace, grace_period_seconds=0)
            if 'web' in request.path:
                return redirect(request.referrer)
            return jsonify({
                "status": 0,
                "message": __("删除完成。查看被删除pod是否完成。")+f"{cluster.get('HOST', request.host).split('|')[-1]}"+conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/')+f"#/search?namespace={namespace}&q={pod_name}",
                "result": {}
            })

        except Exception as e:
            print(e)
            if 'web' in request.path:
                return redirect(request.referrer)
            response = make_response(str(e))
            response.status_code = 500
            return response

    @expose_api(description="打开pod日志界面",url="/web/log/<cluster_name>/<namespace>/<pod_name>", methods=["GET", ])
    @expose_api(description="打开pod日志界面",url="/web/log/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET", ])
    def web_log(self, cluster_name, namespace, pod_name,container_name=None):
        # 验证是否是创建者的pod
        all_clusters = conf.get('CLUSTERS', {})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
        else:
            kubeconfig = None

        k8s_client = K8s(kubeconfig)
        pods = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        if pods:
            pod = pods[0]
            if pod['username']!=g.user.username and not g.user.is_admin():
                return {
                    "message": _('您暂无权限查看此pod日志，仅管理员和创建者可以查看'),
                }
        # 打开iframe页面
        host_url = "//"+ conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host).split('|')[-1]

        if '127.0.0.1' in request.host or 'localhost' in request.host:
            return redirect(host_url+f'{self.route_base}/web/log/{cluster_name}/{namespace}/{pod_name}{("/"+container_name) if container_name else ""}')

        pod_url = host_url + conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, container_name if container_name else pod_name)
        print(pod_url)
        kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')

        k8s_client = K8s(kubeconfig)
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        if pod:
            pod = pod[0]
            if pod['status']=='Running' or pod['status']=='Succeeded':
                flash(__("当前pod状态：")+'%s' % pod['status'], category='warning')
            # if pod['status']=='Failed':
            #     # 获取错误码
            #     flash('当前pod状态：%s' % pod['status'], category='warning')
            else:
                events = k8s_client.get_pod_event(namespace=namespace, pod_name=pod_name)
                if events:
                    event = events[-1]  # 获取最后一个
                    message = event.get('message','')
                    if message:
                        flash(__("当前pod状态：")+'%s，%s' % (pod['status'],message), category='warning')
                else:
                    flash(__("当前pod状态：")+'%s' % pod['status'], category='warning')
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',  # kd-logs-container  :nth-of-type(0)
            "delay": 100,
            "loading": True
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)

    @expose_api(description="打开pod命令行界面",url="/web/debug/<cluster_name>/<namespace>/<pod_name>", methods=["GET", "POST"])
    @expose_api(description="打开pod命令行界面",url="/web/debug/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def web_debug(self, cluster_name, namespace, pod_name, container_name=None):
        # 验证是否是创建者的pod
        all_clusters = conf.get('CLUSTERS', {})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
        else:
            kubeconfig = None

        k8s_client = K8s(kubeconfig)
        pods = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        if pods:
            pod = pods[0]
            if pod['pod_substatus'] in ['Completed', 'Succeeded', 'Failed', 'Error']:
                flash(__('pod已结束，不可进入'), category='warning')
            elif pod['status']!='Running':
                flash(__('pod 尚未启动完成或已结束，刷新当前页面重新进入'), category='warning')
            if pod['username']!=g.user.username and not g.user.is_admin():
                return {
                    "message": _('您暂无权限查看此pod日志，仅管理员和创建者可以查看'),
                }
        # 打开iframe页面
        host_url = "//"+ conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host).split('|')[-1]

        if '127.0.0.1' in request.host or 'localhost' in request.host:
            if container_name:
                return redirect(host_url+f'{self.route_base}/web/debug/{cluster_name}/{namespace}/{pod_name}/{container_name}')
            else:
                return redirect(host_url + f'{self.route_base}/web/debug/{cluster_name}/{namespace}/{pod_name}')
        if container_name:
            pod_url = host_url + conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name, container_name, namespace)
        else:
            pod_url = host_url + conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/') + '#/shell/%s/%s?namespace=%s' % (namespace, pod_name, namespace)
        # print(pod_url)
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',  # 'div.kd-scroll-container.ng-star-inserted',
            "delay": 500,
            "loading": True
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)



    @expose_api(description="打开pod详情界面",url="/web/pod/<cluster_name>/<namespace>/<pod_name>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def web_pod(self, cluster_name, namespace, pod_name):
        # 验证是否是创建者的pod
        all_clusters = conf.get('CLUSTERS', {})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
        else:
            kubeconfig = None

        k8s_client = K8s(kubeconfig)
        pods = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        if pods:
            pod = pods[0]
            if pod['username']!=g.user.username and not g.user.is_admin():
                return {
                    "message": _('您暂无权限查看此pod日志，仅管理员和创建者可以查看'),
                }
        # 打开iframe页面
        host_url = "//"+ conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host).split('|')[-1]

        pod_url = host_url + conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/') + '#/pod/%s/%s?namespace=%s' % (namespace, pod_name, namespace)
        # print(pod_url)
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',  # 'div.kd-scroll-container.ng-star-inserted',
            "delay": 500,
            "loading": True
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)

    @expose_api(description="搜索pod和服务",url="/web/search/<cluster_name>/<namespace>/<search>", methods=["GET", ])
    def web_search(self, cluster_name, namespace, search):
        # 验证是否是创建者的pod
        all_clusters = conf.get('CLUSTERS', {})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
        else:
            kubeconfig = None

        k8s_client = K8s(kubeconfig)
        # 查询pod
        pods = k8s_client.v1.list_namespaced_pod(namespace=namespace).items or []
        pods = [k8s_client.pod_model2dict(pod) for pod in pods if search in pod.metadata.name]
        pods = sorted(pods, key=lambda pod: pod['start_time'])
        if not g.user.is_admin():
            pods = [pod for pod in pods if pod['username'] == g.user.username]
        # 查询服务
        services = k8s_client.v1.list_namespaced_service(namespace=namespace).items or []
        services = [service for service in services if search in service.metadata.name]
        if not g.user.is_admin():
            services = [service for service in services if k8s_client.get_username(service.metadata.labels) == g.user.username]

        data_pods=[]
        for pod in pods:
            if 'main' in pod['containers']:
                container_name = 'main'
            else:
                container_name = pod['containers'][-1]
            data_pod = [
                Markup(status_icon.get(pod['pod_substatus'],default_status_icon)),
                Markup(f'<a href="/k8s/web/pod/{cluster_name}/{namespace}/{pod["name"]}">{pod["name"]}</a>'),
                Markup('<br>'.join([f'<div class="chip">{image}</div>' for image in pod['images']])),
                pod['host_ip'],
                pod['pod_substatus'],
                pod['restart_count'],
                Markup('<div style="min-width:120px">'+humanize.naturaltime(datetime.datetime.now() - pod['start_time'])+"</div>"),
                Markup(f'<div style="min-width:180px"><a href="/k8s/web/log/{cluster_name}/{namespace}/{pod["name"]}/{container_name}">日志</a> | <a href="/k8s/web/debug/{cluster_name}/{namespace}/{pod["name"]}/{container_name}">进入</a> | <a href="{conf.get("GRAFANA_TASK_PATH")}{pod["name"]}">监控</a> | <a href="/k8s/web/delete/pod/{cluster_name}/{namespace}/{pod["name"]}">删除</a></div>' )
            ]
            data_pods.append({
                "one":data_pod,
                "errors":pod['message'],
            })

        data_services = []
        for service in services:
            data_service = [
                Markup(status_icon['Running']),
                Markup(service.metadata.name),
                Markup(service.spec.cluster_ip or ""),
                Markup('<br>'.join([f'{service.metadata.name}.{namespace}:{port.port}' for port in service.spec.ports])) if service.spec.ports else '',
                Markup('<br>'.join([f'<a href="http://{service.spec.external_i_ps[0]}:{port.port}">{service.spec.external_i_ps[0]}:{port.port}</a>' for port in service.spec.ports]) if service.spec.ports and service.spec.external_i_ps else ''),
                humanize.naturaltime(datetime.datetime.now(tz=tzutc()) - service.metadata.creation_timestamp),
            ]
            error = []
            data_services.append({
                "one": data_service,
                "errors": error,
            })

        data=[]

        data.append(
            {
                "label": "Pods",
                "columns": ["", "Name", "Images", "Node", "Status", 'Restarts', 'Created', "Ops"],
                "data": data_pods,
            }
        )

        data.append(
            {
                "label": "Services",
                "columns": ["", "Name", "Cluster ip", 'Internal Endpoints', 'External Endpoints', 'Created'],
                "data": data_services,
                "errors": ""
            }
        )
        return self.render_template('pods.html', data=data)


    @expose_api(description="搜索pod和服务",url="/web1/search/<cluster_name>/<namespace>/<search>", methods=["GET", ])
    def web1_search(self, cluster_name, namespace, search):
        host_url = "//" + conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host).split('|')[-1]
        if '127.0.0.1' in request.host or 'localhost' in request.host:
            return redirect(host_url+f'{self.route_base}/web/search/{cluster_name}/{namespace}/{search}')

        search = search[:50]
        pod_url = host_url+conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/') + "#/search?namespace=%s&q=%s" % (namespace, search)
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',  # kd-logs-container  :nth-of-type(0)
            "delay": 500,
            "loading": True
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)

appbuilder.add_api(K8s_View)
