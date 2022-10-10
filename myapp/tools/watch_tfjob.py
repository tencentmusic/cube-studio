

import time,datetime,logging,os,sys
import asyncio
from kubernetes import client
from kubernetes import watch
from os import path
import json
import requests
from myapp.utils.py.py_k8s import check_status_time,K8s
from sqlalchemy.exc import InvalidRequestError,OperationalError
import pysnooper
import myapp
import math
from myapp import app, db, security_manager
from myapp.models.model_job import (
    Tfjob,
    Task
)
from myapp.utils.celery import session_scope
from myapp.project import push_admin,push_message
from myapp.models.model_job import Pipeline,Workflow
import pymysql
conf=app.config

from myapp.utils.py.py_prometheus import Prometheus

prometheus = Prometheus(conf.get('PROMETHEUS',''))

cluster=os.getenv('ENVIRONMENT','').lower()
if not cluster:
    print('no cluster %s'%cluster)
    exit(1)
else:
    clusters = conf.get('CLUSTERS',{})
    if clusters and cluster in clusters:
        kubeconfig = clusters[cluster].get('KUBECONFIG','')
        K8s(kubeconfig)
    else:
        print('no kubeconfig in cluster %s' % cluster)
        exit(1)

# 推送消息
# @pysnooper.snoop()
def deliver_message(tfjob):
    if not tfjob:
        return
    receivers = tfjob.username.split(',')
    receivers = [receiver.strip() for receiver in receivers]
    if not receivers:
        print('no receivers')
        return

    info_json = json.loads(tfjob.info_json)
    # print(info_json,experiments.status)
    if tfjob.status in info_json['alert_status'] and tfjob.status not in info_json['has_push']:
        receivers=list(set(receivers))
        # data = {
        #     "Sender": sender,
        #     "Rcptto":receivers,
        # }
        workflow_name = info_json.get('workflow_name','')
        hp_name = info_json.get('hp_name', '')
        if workflow_name:
            message = "tfjob: %s \nworkflow: %s \nnamespace: %s\nstatus: %s \ntime: %s" % (tfjob.name,workflow_name,tfjob.namespace,tfjob.status,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        elif hp_name:
            message = "tfjob: %s \nhp: %s(%s) \nnamespace: %s\nstatus: %s \ntime: %s" % (tfjob.name,info_json.get('hp_name',''),info_json.get('describe',''),tfjob.namespace,tfjob.status,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            message = "tfjob: %s \nnamespace: %s\nstatus: %s \ntime: %s" % (tfjob.name,tfjob.namespace,tfjob.status,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        if message:
            push_message(receivers,message)


# @pysnooper.snoop()
def check_has_push(crd,dbsession):
    # 可能是workflow启动的或者是hp启动的
    workflow_name = crd['labels'].get('workflow-name','')
    hp_name = crd['labels'].get('hp-name', '')
    username = crd['username']
    alert_status = ''
    # 如果是从workflow中创建的
    if workflow_name:
        pipeline = dbsession.query(Pipeline).filter_by(name=workflow_name).first()
        if pipeline and pipeline.alert_status:
            alert_status = pipeline.alert_status
        print("tf %s from workflow_name %s,user %s,status %s" % (crd['name'],workflow_name,crd['username'],crd['status']))

    if hp_name:
        hp = dbsession.query(Hyperparameter_Tuning).filter_by(name=hp_name).first()
        if hp and hp.alert_status:
            alert_status = hp.alert_status

        print("tf %s from hp %s,user %s,status %s" % (crd['name'], workflow_name, crd['username'], crd['status']))

    # print("%s status %s"%(crd['name'], crd['status']))
    alert_status='Pending'   # 这里写死，就是相当于必须且仅Pending告警

    info_json={
        "workflow_name":workflow_name,
        "hp_name":hp_name,
        "alert_status": alert_status,
        "has_push":''
    }
    # print(crd['name'],crd['namespace'])
    tfjob = dbsession.query(Tfjob).filter(Tfjob.name==crd['name']).filter(Tfjob.namespace==crd['namespace']).first()
    if tfjob:
        print('exist tfjob')
        if tfjob.info_json:
            exist_info_json = json.loads(tfjob.info_json)
            info_json['has_push']=exist_info_json.get('has_push','')

        tfjob.create_time = crd['create_time']
        tfjob.status = crd['status']
        tfjob.annotations = json.dumps(crd['annotations'],indent=4,ensure_ascii=False)
        tfjob.labels = json.dumps(crd['labels'],indent=4,ensure_ascii=False)
        tfjob.spec = json.dumps(crd['spec'],indent=4,ensure_ascii=False),
        tfjob.status_more = json.dumps(crd['status_more'],indent=4,ensure_ascii=False)
        tfjob.username = crd['username']
        tfjob.info_json = json.dumps(info_json,indent=4,ensure_ascii=False)
        dbsession.commit()

        if crd['status'] in info_json['alert_status'] and crd['status'] not in info_json['has_push']:
            return False,tfjob
        else:
            return True,tfjob
    else:
        print('new tfjob')
        # crd['status_more']={}
        # crd['spec']={}
        tfjob = Tfjob(name=crd['name'],namespace=crd['namespace'],create_time=crd['create_time'],
                            status=crd['status'],
                            annotations=json.dumps(crd['annotations'],indent=4,ensure_ascii=False),
                            labels=json.dumps(crd['labels'],indent=4,ensure_ascii=False),
                            spec=json.dumps(crd['spec'],indent=4,ensure_ascii=False),
                            status_more=json.dumps(crd['status_more'],indent=4,ensure_ascii=False),
                            username=username,
                            info_json=json.dumps(info_json,indent=4,ensure_ascii=False))

        dbsession.add(tfjob)
        dbsession.commit()
        return False,tfjob


#
# # 推送修改通知
# @pysnooper.snoop()
# def push_resource_rec(task,dbsession):
#     task_monitorings = json.loads(task.monitoring).get('tfjob',[])
#     if len(task_monitorings)>9:
#         max_cpu = 0
#         max_memory=0
#         init_message = 'pipeline(%s)中分布式训练%s，推荐资源如下，自行修改:\n' % (task.pipeline.describe,task.label)
#         message = init_message
#         # tfjob_src_mem=re.match(task.args.match("memory": "32G",))
#         for task_monitoring in task_monitorings:
#             if float(task_monitoring.get('cpu',0))>max_cpu:
#                 max_cpu = float(task_monitoring.get('cpu',0))
#             if float(task_monitoring.get('memory', 0)) > max_memory:
#                 max_memory = float(task_monitoring.get('memory', 0))
#         if max_cpu:
#             rec_cpu = math.ceil(max_cpu*1.4)
#             if rec_cpu>150:
#                 rec_cpu=150
#             if rec_cpu!=int(task.resource_cpu):
#                 message += "task(%s)，原申请cpu:%s，近10次最大使用cpu:%s，建议申请值:%s\n" % (task.label,task.resource_cpu, max_cpu, rec_cpu)
#                 task.resource_cpu = str(rec_cpu)
#         if max_memory:
#             rec_memory = math.ceil(max_memory*1.4)
#             if rec_memory>350:
#                 rec_memory=350
#             if rec_memory!=int(task.resource_memory.replace('G','').replace('M','')):
#                 message += "task(%s)，原申请mem:%s，近10次最大使用mem:%s(G)，建议申请值:%s\n" % (task.label,task.resource_memory, max_memory, str(rec_memory)+"G")
#                 task.resource_memory = str(rec_memory)+"G"
#
#         dbsession.commit()
#         if message!=init_message:
#             push_message([task.pipeline.created_by.username],message)



# @pysnooper.snoop()
def save_monitoring(tfjob,dbsession):
    try:
        if tfjob.status=='Succeeded':
            task_id = json.loads(tfjob.labels).get('task-id','')
            if task_id:
                task = dbsession.query(Task).filter_by(id=int(task_id)).first()
                metrics = prometheus.get_resource_metric(tfjob.name, namespace='pipeline')
                monitoring = json.loads(task.monitoring) if task and task.monitoring else {}

                tfjob_monitoring = monitoring.get('tfjob', [])
                if metrics:
                    tfjob_monitoring.append({
                        "cpu": metrics.get('cpu', ''),
                        "memory": metrics.get('memory', ''),
                        "name": tfjob.name,
                        "update_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                # 清理监控记录
                tfjob_monitoring_new = []
                for metric in tfjob_monitoring:
                    # 采集结果不对的，和采集结果太久远的都清理掉
                    if float(metric.get('cpu', 0)) > 0.1 and float(metric.get('memory', 0)) > 0.1 and metric['update_time'] > (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S'):
                        tfjob_monitoring_new.append(metric)

                if len(tfjob_monitoring_new) > 10:
                    del tfjob_monitoring_new[0]

                monitoring_new = {}
                monitoring_new['task'] = monitoring.get('task', [])
                monitoring_new['tfjob'] = tfjob_monitoring_new

                print(monitoring_new)
                if task:
                    task.monitoring = json.dumps(monitoring_new,ensure_ascii=False,indent=4)
                    dbsession.commit()
                    # print(pods)

                    # push_resource_rec(task, dbsession)

    except Exception as e:
        print(e)



# @pysnooper.snoop()
def save_history(tfjob,dbsession):
    info_json = json.loads(tfjob.info_json)
    if info_json['has_push']:
        if not tfjob.status in info_json['has_push']:
            info_json['has_push'] += ',' + tfjob.status
    else:
        info_json['has_push'] = tfjob.status
    tfjob.info_json = json.dumps(info_json, indent=4, ensure_ascii=False)
    dbsession.commit()


# @pysnooper.snoop()
def check_crd_exist(group,version,namespace,plural,name):
    exist_crd = client.CustomObjectsApi().get_namespaced_custom_object(group,version,namespace,plural,name)
    return exist_crd


@pysnooper.snoop()
def deal_event(event,crd_info,namespace):
    with session_scope(nullpool=True) as dbsession:
        try:
            crd_object = event['object']
            exist_crd = check_crd_exist(group=crd_info['group'], version=crd_info["version"], namespace=namespace,
                                        plural=crd_info["plural"], name=crd_object['metadata']['name'])
            if not exist_crd:
                print('not exist')
                return

            status = ''
            if 'status' in crd_object:
                if 'conditions' in crd_object['status']:
                    if len(crd_object['status']['conditions']) > 0:
                        if 'type' in crd_object['status']['conditions'][-1]:
                            status = crd_object['status']['conditions'][-1]['type']

            creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
            creat_time = (datetime.datetime.strptime(creat_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')

            back_object = {
                'username': '',
                "name": crd_object['metadata']['name'],
                "namespace": crd_object['metadata']['namespace'] if 'namespace' in crd_object['metadata'] else '',
                "annotations": crd_object['metadata'].get('annotations', {}),
                "labels": crd_object['metadata'].get('labels', {}),
                "spec": crd_object['spec'],
                "create_time": creat_time,
                "status": status,
                "status_more": check_status_time(crd_object['status']) if 'status' in crd_object else {}
            }
            if 'run-rtx' in back_object['labels']:
                back_object['username'] = back_object['labels']['run-rtx']
            elif 'upload-rtx' in back_object:
                back_object['username'] = back_object['labels']['upload-rtx']

            has_push, crd_model = check_has_push(back_object,dbsession)
            if not has_push:
                try:
                    deliver_message(crd_model)
                except Exception as e1:
                    print('push fail:', e1)
                    push_admin(str(e1))
            save_history(crd_model,dbsession)
            save_monitoring(crd_model,dbsession)
        except Exception as e:
            print(e)

@pysnooper.snoop()
def listen_crd():
    crd_info = conf.get('CRD_INFO')['tfjob']
    namespace = conf.get('PIPELINE_NAMESPACE')
    w = watch.Watch()
    print('begin listen')
    while(True):
        try:
            for event in w.stream(client.CustomObjectsApi().list_namespaced_custom_object, group=crd_info['group'],
                                  version=crd_info['version'],
                                  namespace=namespace, plural=crd_info['plural'], pretty='true'):

                if event['type']=='ADDED' or event['type']=='MODIFIED':  # ADDED  MODIFIED DELETED
                    deal_event(event,crd_info,namespace)
                elif event['type']=='ERROR':
                    w = watch.Watch()
                    time.sleep(60)

        except Exception as ee:
            print(ee)

# 不能使用异步io，因为stream会阻塞
if __name__=='__main__':
    listen_crd()

