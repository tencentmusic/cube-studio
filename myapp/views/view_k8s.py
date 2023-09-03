import json

import pysnooper, os

from myapp import app, conf
from myapp.utils.py.py_k8s import K8s, K8SStreamThread
from flask import g, flash, request, render_template, send_from_directory, send_file, make_response, Markup, jsonify
import flask_socketio
import datetime, time
from myapp import app, appbuilder, db, event_logger
from .base import BaseMyappView
from flask_appbuilder import CompactCRUDMixin, expose


class K8s_View(BaseMyappView):
    route_base = '/k8s'

    # 打开pod日志界面
    @expose("/watch/log/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET", ])
    def watch_log(self, cluster_name, namespace, pod_name, container_name):
        data = {
            "url": '/k8s/stream/log',
            "server_event_name": "server_event_name",
            "user_event_name": cluster_name + "_" + namespace + "_" + pod_name + "_" + container_name,
            "cluster_name": cluster_name,
            "namespace_name": namespace,
            "pod_name": pod_name,
            "container_name": container_name,
        }
        print(data)
        return self.render_template('log.html', data=data)

    # from myapp import socketio
    # # 实时获取pod日志
    # @socketio.on("k8s_stream_log",namespace='/k8s/stream/log')
    # # @pysnooper.snoop()
    # def stream_log(*args,**kwargs):
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
    @expose("/watch/exec/<cluster_name>/<namespace>/<pod_name>/<container_name>")
    def watch_exec(self, cluster_name, namespace, pod_name, container_name):
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
    @expose("/download/log/<cluster_name>/<namespace>/<pod_name>")
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
    @expose("/read/log/<cluster_name>/<namespace>/<pod_name>")
    @expose("/read/log/<cluster_name>/<namespace>/<pod_name>/<container>")
    @expose("/read/log/<cluster_name>/<namespace>/<pod_name>/<container>/<tail>")
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
    @expose("/read/pod/<cluster_name>/<namespace>/<pod_name>")
    def read_pod(self, cluster_name, namespace, pod_name):
        try:
            all_clusters = conf.get('CLUSTERS', {})
            if cluster_name in all_clusters:
                kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')
            else:
                kubeconfig = None

            k8s = K8s(kubeconfig)
            pods = k8s.get_pods(namespace=namespace, pod_name=pod_name)
            if pods:
                pod = pods[0]
                pod['events'] = k8s.get_pod_event(namespace=namespace, pod_name=pod_name)
                return jsonify({
                    "status": 0,
                    "message": "",
                    "result": pod
                })
            else:
                return jsonify({
                    "status": 1,
                    "message": "pod不存在",
                    "result": {}
                })

        except Exception as e:
            print(e)
            response = make_response(str(e))
            response.status_code = 500
            return response

    terminating_pods = {
        "time": None,
        "data": {}
    }

    # 返回获取terminating不掉的pod的信息
    @expose("/read/pod/terminating")
    # @pysnooper.snoop()
    def read_terminating_pod(self, namespace='service'):
        try:
            if not self.terminating_pods['time'] or (datetime.datetime.now() - self.terminating_pods['time']).total_seconds()>200:
                clusters = conf.get('CLUSTERS', {})
                for cluster_name in clusters:
                    try:
                        self.terminating_pods['data'][cluster_name] = {}  # 重置，重新查询
                        cluster = clusters[cluster_name]
                        k8s_client = K8s(cluster.get('KUBECONFIG', ''))

                        events = [item.to_dict() for item in k8s_client.v1.list_namespaced_event(namespace=namespace).items]  # ,field_selector=f'source.host={ip}'
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
                                    self.terminating_pods['data'][cluster_name][pod_name] = {
                                        "namespace": namespace,
                                        "host": host,
                                        "host"
                                        "begin": event['time'],
                                        "username": pods_dict.get(pod_name, {}).get("username", ''),
                                        "label": pods_dict.get(pod_name, {}).get("label", '')
                                    }
                    except Exception as e:
                        print(e)
                self.terminating_pods['time'] = datetime.datetime.now()

            return jsonify(self.terminating_pods['data'])

        except Exception as e:
            print(e)
            response = make_response(str(e))
            response.status_code = 500
            return response

    # 强制删除pod的信息
    @expose("/delete/pod/<cluster_name>/<namespace>/<pod_name>")
    def delete_pod(self, cluster_name, namespace, pod_name):
        try:
            all_clusters = conf.get('CLUSTERS', {})
            cluster = all_clusters[cluster_name]
            kubeconfig = cluster.get('KUBECONFIG', '')

            from myapp.utils.core import run_shell
            command = f'kubectl delete pod {pod_name} -n {namespace} --force --grace-period=0 '
            if kubeconfig:
                command += f' --kubeconfig {kubeconfig}'

            status = run_shell(command)
            return jsonify({
                "status": 0,
                "message": f"删除完成。查看被删除pod是否完成。http://{cluster.get('HOST', request.host)}"+conf.get('K8S_DASHBOARD_CLUSTER','/k8s/dashboard/cluster/')+f"#/search?namespace={namespace}&q={pod_name}",
                "result": {}
            })

        except Exception as e:
            print(e)
            response = make_response(str(e))
            response.status_code = 500
            return

    @expose("/web/log/<cluster_name>/<namespace>/<pod_name>", methods=["GET", ])
    @expose("/web/log/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET", ])
    def web_log(self, cluster_name, namespace, pod_name,container_name=None):
        from myapp.utils.py.py_k8s import K8s
        all_clusters = conf.get('CLUSTERS', {})
        host_url = "http://" + conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host)
        pod_url = host_url + conf.get('K8S_DASHBOARD_CLUSTER') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, container_name if container_name else pod_name)
        print(pod_url)
        kubeconfig = all_clusters[cluster_name].get('KUBECONFIG', '')

        k8s = K8s(kubeconfig)
        pod = k8s.get_pods(namespace=namespace, pod_name=pod_name)
        if pod:
            pod = pod[0]
            if pod['status']=='Running' or pod['status']=='Succeeded':
                flash('当前pod状态：%s' % pod['status'], category='warning')
            # if pod['status']=='Failed':
            #     # 获取错误码
            #     flash('当前pod状态：%s' % pod['status'], category='warning')
            else:
                events = k8s.get_pod_event(namespace=namespace, pod_name=pod_name)
                if events:
                    event = events[-1]  # 获取最后一个
                    message = event.get('message','')
                    if message:
                        flash('当前pod状态：%s，%s' % (pod['status'],message), category='warning')
                else:
                    flash('当前pod状态：%s' % pod['status'], category='warning')
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',  # kd-logs-container  :nth-of-type(0)
            "delay": 1000,
            "loading": True
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)

    @expose("/web/debug/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def web_debug(self, cluster_name, namespace, pod_name, container_name):

        host_url = "http://" + conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host)
        pod_url = host_url + conf.get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name, container_name, namespace)
        print(pod_url)
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',  # 'div.kd-scroll-container.ng-star-inserted',
            "delay": 1000,
            "loading": True
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)


    @expose("/web/search/<cluster_name>/<namespace>/<search>", methods=["GET", ])
    def web_search(self, cluster_name, namespace, search):
        host_url = "http://" + conf.get("CLUSTERS", {}).get(cluster_name, {}).get("HOST", request.host)
        search = search[:50]
        pod_url = host_url+conf.get('K8S_DASHBOARD_CLUSTER') + "#/search?namespace=%s&q=%s" % (namespace, search)
        data = {
            "url": pod_url,
            "target": 'div.kd-scroll-container',  # kd-logs-container  :nth-of-type(0)
            "delay": 1000,
            "loading": True
        }
        # 返回模板
        if cluster_name == conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)

appbuilder.add_api(K8s_View)
