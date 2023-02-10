import json

import pysnooper

from myapp import app,conf
from myapp.utils.py.py_k8s import K8s,K8SStreamThread
from flask import g,flash,request,render_template,send_from_directory,send_file,make_response,Markup,jsonify
import flask_socketio
import datetime,time

# 打开pod日志界面
@app.route("/k8s/web/log/<cluster_name>/<namespace>/<pod_name>/<container_name>", methods=["GET",])
def web_log(cluster_name,namespace,pod_name,container_name):
    data = {
        "url":'/k8s/stream/log',
        "server_event_name":"server_event_name",
        "user_event_name":cluster_name+"_"+namespace+"_"+pod_name+"_"+container_name,
        "cluster_name": cluster_name,
        "namespace_name": namespace,
        "pod_name": pod_name,
        "container_name": container_name,
    }
    print(data)
    return render_template('log.html',data=data)

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
@app.route("/k8s/web/exec/<cluster_name>/<namespace>/<pod_name>/<container_name>")
@pysnooper.snoop()
def web_exec(cluster_name,namespace,pod_name,container_name):
    data = {
        "ws_url":f"/k8s/stream/exec/{cluster_name}/{namespace}/{pod_name}/{container_name}"
    }
    # 返回模板
    return render_template('terminal.html', data=data)

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
@app.route("/k8s/download/log/<cluster_name>/<namespace>/<pod_name>")
def download_log(cluster_name,namespace,pod_name):
    try:
        all_clusters = conf.get('CLUSTERS',{})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG','')
        else:
            kubeconfig = None

        k8s = K8s(kubeconfig)
        logs = k8s.download_pod_log(namespace=namespace, name=pod_name)
        file = open(pod_name,mode='w')
        file.write(logs)
        file.close()
        response = make_response(send_file(pod_name, as_attachment=True, conditional=True))
        return response
    except Exception as e:
        print(e)
        return str(e)

# 返回获取pod日志
@app.route("/k8s/read/log/<cluster_name>/<namespace>/<pod_name>")
@app.route("/k8s/read/log/<cluster_name>/<namespace>/<pod_name>/<container>")
@app.route("/k8s/read/log/<cluster_name>/<namespace>/<pod_name>/<container>/<tail>")
def read_log(cluster_name,namespace,pod_name,container=None,tail=None):
    try:
        all_clusters = conf.get('CLUSTERS',{})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG','')
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
            logs = k8s.download_pod_log(namespace=namespace, name=pod_name, container=container,tail_lines=tail)

        return logs
    except Exception as e:
        print(e)
        return str(e)



# 返回获取pod的信息
@app.route("/k8s/read/pod/<cluster_name>/<namespace>/<pod_name>")
def read_pod(cluster_name,namespace,pod_name):
    try:
        all_clusters = conf.get('CLUSTERS',{})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name].get('KUBECONFIG','')
        else:
            kubeconfig = None

        k8s = K8s(kubeconfig)
        pods = k8s.get_pods(namespace=namespace, pod_name=pod_name)
        if pods:
            pod = pods[0]
            pod['events'] = k8s.get_pod_event(namespace=namespace,pod_name=pod_name)
            return jsonify({
                "status":0,
                "message":"",
                "result":pod
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