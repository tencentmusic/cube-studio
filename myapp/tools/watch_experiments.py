

import time,datetime,logging,os,sys
import asyncio
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes import watch
from os import path
import json
import requests
from sqlalchemy.exc import InvalidRequestError,OperationalError
import pysnooper
import myapp
from myapp import app, db, security_manager
from myapp.models.model_katib import (
    Experiments,
    Hyperparameter_Tuning
)
from myapp.utils.py.py_k8s import check_status_time,K8s
from myapp.utils.celery import session_scope
from myapp.project import push_admin,push_message
import pymysql
conf=app.config

cluster=os.getenv('ENVIRONMENT','').lower()
if not cluster:
    print('no cluster %s'%cluster)
    exit(1)
else:
    clusters = conf.get('CLUSTERS',{})
    if clusters and cluster in clusters:
        kubeconfig = clusters[cluster].get('KUBECONFIG','')
        k8s_client = K8s(kubeconfig)
        # k8s_config.kube_config.load_kube_config(config_file=kubeconfig)
    else:
        print('no kubeconfig in cluster %s' % cluster)
        exit(1)


# 推送微信消息
@pysnooper.snoop()
def deliver_message(experiments):
    if not experiments:
        return
    receivers = experiments.username.split(',')
    receivers = [receiver.strip() for receiver in receivers]
    if not receivers:
        print('no receivers')
        return

    info_json = json.loads(experiments.info_json)
    # print(info_json,experiments.status)
    if experiments.status in info_json['alert_status'] and experiments.status not in info_json['has_push']:
        receivers=list(set(receivers))
        # data = {
        #     "Sender": sender,
        #     "Rcptto":receivers,
        # }
        message = "experiments: %s \nhp: %s(%s) \nnamespace: %s\nstatus: % s \ntime: %s" % (experiments.name,info_json.get('hp_name',''),info_json.get('describe',''),experiments.namespace,experiments.status,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        if message:
            push_message(receivers,message)



@pysnooper.snoop()
def check_has_push(crd,dbsession):

    hp_name = crd['labels'].get('hp-name','')
    hp_describe = crd['labels'].get('hp-describe','')

    alert_status = ''
    hp = dbsession.query(Hyperparameter_Tuning).filter_by(name=hp_name).first()
    if hp and hp.alert_status:
        alert_status = hp.alert_status
    username = crd['username']
    print("Event: % s %s %s %s %s" % (crd['name'],hp_describe,hp_name,crd['username'],crd['status']))
    # print("%s status %s"%(crd['name'], crd['status']))
    info_json={
        "hp_name":hp_name,
        "hp_describe":hp_describe,
        "alert_status": alert_status,
        "has_push":''
    }
    # print(crd['name'],crd['namespace'])
    experiments = dbsession.query(Experiments).filter(Experiments.name==crd['name']).filter(Experiments.namespace==crd['namespace']).first()
    if experiments:
        print('exist experiments')
        if experiments.info_json:
            exist_info_json = json.loads(experiments.info_json)
            info_json['has_push']=exist_info_json.get('has_push','')

        experiments.create_time = crd['create_time']
        experiments.status = crd['status']
        experiments.annotations = json.dumps(crd['annotations'],indent=4,ensure_ascii=False)
        experiments.labels = json.dumps(crd['labels'],indent=4,ensure_ascii=False)
        experiments.spec = json.dumps(crd['spec'],indent=4,ensure_ascii=False),
        experiments.status_more = json.dumps(crd['status_more'],indent=4,ensure_ascii=False)
        experiments.username = crd['username']
        experiments.info_json = json.dumps(info_json,indent=4,ensure_ascii=False)
        dbsession.commit()

        if crd['status'] in info_json['alert_status'] and crd['status'] not in info_json['has_push']:
            return False,experiments
        else:
            return True,experiments
    else:
        print('new experiments')
        # crd['status_more']={}
        # crd['spec']={}
        experiments = Experiments(name=crd['name'],namespace=crd['namespace'],create_time=crd['create_time'],
                            status=crd['status'],
                            annotations=json.dumps(crd['annotations'],indent=4,ensure_ascii=False),
                            labels=json.dumps(crd['labels'],indent=4,ensure_ascii=False),
                            spec=json.dumps(crd['spec'],indent=4,ensure_ascii=False),
                            status_more=json.dumps(crd['status_more'],indent=4,ensure_ascii=False),
                            username=username,
                            info_json=json.dumps(info_json,indent=4,ensure_ascii=False))

        dbsession.add(experiments)
        dbsession.commit()
        return False,experiments

@pysnooper.snoop()
def save_history(experiments,dbsession):
    info_json = json.loads(experiments.info_json)
    if info_json['has_push']:
        if not experiments.status in info_json['has_push']:
            info_json['has_push'] += ',' + experiments.status
    else:
        info_json['has_push'] = experiments.status
    experiments.info_json = json.dumps(info_json, indent=4, ensure_ascii=False)
    dbsession.commit()


@pysnooper.snoop()
def check_crd_exist(group,version,namespace,plural,name):
    client = k8s_client.CustomObjectsApi()
    exist_crd = client.get_namespaced_custom_object(group,version,namespace,plural,name)
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
                "status_more": crd_object['status'] if 'status' in crd_object else {}
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

        except Exception as e:
            print(e)


@pysnooper.snoop()
def listen_crd():
    crd_info = conf.get('CRD_INFO')['experiment']
    namespace = conf.get('KATIB_NAMESPACE')
    w = watch.Watch()
    # k8s_client.CustomObjectsApi()
    print('begin listen')
    while(True):
        try:
            for event in w.stream(k8s_client.CustomObjectsApi().list_namespaced_custom_object, group=crd_info['group'],
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

