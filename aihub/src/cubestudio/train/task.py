import logging
import os
import requests
import pysnooper
import json
import re,datetime,time,os,sys,io
import inspect
from cubestudio.request.model import Model

class Task(Model):
    path = '/task_modelview/api'
    id=None

    # 启动
    def run(self,until='Running'):
        begin_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        url = self.client.req(self.client.path+f"/run/{self.id}")
        print(f'任务启动成功，日志地址：{url}')
        exist_message = []
        status = self.status()
        while (status != until):
            running_pod = self.pod()
            events = running_pod['events']
            now_status = running_pod['status']
            # 有新消息要打印
            for event in events:
                message = f'"时间："{event["time"]} ，类型：{event["type"]} ，原因：{event["reason"]} ，消息：{event["message"]}'
                if message not in exist_message and event['time']>begin_time:
                    print(message,flush=True)
                    exist_message.append(message)
            # 状态变化要打印
            if now_status!=status:
                status = now_status
                print(status,flush=True)
            time.sleep(5)

    def status(self):
        running_pod_name = "run-" + self.pipeline.name.replace('_', '-') + "-" + self.name.replace('_', '-')
        result = self.client.req(f"/k8s/read/pod/dev/pipeline/{running_pod_name}")
        status = result.get("result",{}).get("status",'未知')
        # print(status)
        return status

    def debug(self):
        pass

    def log(self,follow=False):

        running_pod_name = "run-" + self.pipeline.name.replace('_', '-') + "-" + self.name.replace('_', '-')
        if follow:
            status = 'Running'
            while status=='Running':
                status = self.status()
                logs = self.client.req(f"/k8s/read/log/dev/pipeline/{running_pod_name}/{running_pod_name}/5s")
                print(logs,flush=True)
                time.sleep(5)

            # from cubestudio.request.model_client import client, HOST, USERNAME
            # client = socketio.Client()
            # try:
            #     user_event_name = USERNAME+"_sdk_task"
            #
            #     client.connect(HOST,namespaces=['/k8s/stream/log'])
            #
            #     # client.on(user_event_name, on_response)
            #
            #     client.emit('k8s_stream_log', data = json.dumps({
            #         "cluster_name":"dev",
            #         "namespace_name":"pipeline",
            #         "pod_name":running_pod_name,
            #         "container_name":running_pod_name,
            #         "user_event_name":user_event_name
            #     }),namespace='/k8s/stream/log')
            #
            #     @client.on(user_event_name)
            #     def on_message(args):
            #         print(args)
            #         print('on_server_response', args['data'])
            #     client.wait()
            #     # time.sleep(10)
            # finally:
            #     client.disconnect()
        else:
            logs = self.client.req(f"/k8s/read/log/dev/pipeline/{running_pod_name}")
            print(logs)
            return logs

    def monitoring(self):
        pass

    def pod(self):
        running_pod_name = "run-" + self.pipeline.name.replace('_', '-') + "-" + self.name.replace('_', '-')
        running_pod_result = self.client.req(f"/k8s/read/pod/dev/pipeline/{running_pod_name}")
        running_pod_result = running_pod_result.get("result", {})
        running_pod_result['message'] = [f'"时间："{event["time"]} ，类型：{event["type"]} ，原因：{event["reason"]} ，消息：{event["message"]}' for event in running_pod_result.get('events',[])]
        return running_pod_result

    def stop(self):
        self.client.req(self.client.path + f"/clear/{self.id}")



class Repository(Model):
    path='/repository_modelview/api'

class Images(Model):
    path='/images_modelview/api'

class Job_Template(Model):
    path = '/job_template_fab_modelview/api'

class Project(Model):
    path='/project_modelview/api'

class Pipeline(Model):
    path='/pipeline_modelview/api'

class Workflow(Model):
    path='/workflow_modelview/api'

class MyUser(Model):
    path='/users/api'

User=MyUser
