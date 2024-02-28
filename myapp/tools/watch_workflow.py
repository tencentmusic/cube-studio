import pysnooper
import time, datetime, os
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from kubernetes import client
from kubernetes import watch
import json
import math
from myapp.utils.py.py_k8s import check_status_time, K8s
from myapp.utils.py.py_prometheus import Prometheus
from myapp.project import push_message
from myapp import app
from myapp.models.model_job import (
    Pipeline,
    Workflow,
    Task,
    RunHistory
)

from myapp.utils.celery import session_scope

conf = app.config
prometheus = Prometheus(conf.get('PROMETHEUS', ''))

cluster = os.getenv('ENVIRONMENT', '').lower()
if not cluster:
    print('no cluster %s' % cluster)
    exit(1)
else:
    clusters = conf.get('CLUSTERS', {})
    if clusters and cluster in clusters:
        kubeconfig = clusters[cluster].get('KUBECONFIG', '')
        K8s(kubeconfig)
        # k8s_config.kube_config.load_kube_config(config_file=kubeconfig)
    else:
        print('no kubeconfig in cluster %s' % cluster)
        exit(1)


# 推送微信消息
# @pysnooper.snoop()
def deliver_message(workflow, dbsession):
    if not workflow:
        return

    receivers = workflow.username.split(',')
    receivers = [receiver.strip() for receiver in receivers]

    pipeline_id = json.loads(workflow.labels).get("pipeline-id", '')
    if pipeline_id and int(pipeline_id) > 0:
        pipeline = dbsession.query(Pipeline).filter_by(id=int(pipeline_id)).first()
        alert_user = pipeline.alert_user.split(',') if pipeline.alert_user else []
        alert_user = [user.strip() for user in alert_user if user.strip()]
        receivers += alert_user

    if not receivers:
        print('no receivers')
        return

    info_json = json.loads(workflow.info_json)
    # print(info_json,workflow.status)
    if workflow.status in info_json['alert_status'] and workflow.status not in info_json['has_push']:
        receivers = list(set(receivers))
        # data = {
        #     "Sender": sender,
        #     "Rcptto":receivers,
        # }
        status_more = json.loads(workflow.status_more)
        start_time = status_more.get('startedAt','').replace('T',' ').replace('Z','') if status_more.get('startedAt','') else ''
        if start_time:
            start_time = (datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=0)).strftime('%Y-%m-%d %H:%M:%S')

        finish_time = status_more.get('finishedAt', '').replace('T', ' ').replace('Z', '') if status_more.get('finishedAt','') else ''
        if finish_time:
            finish_time = (datetime.datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=0)).strftime(
                '%Y-%m-%d %H:%M:%S')
        help_url='http://%s/pipeline_modelview/api/web/pod/%s'%(conf.get('HOST'),pipeline_id)
        message = "workflow: %s \npipeline: %s(%s) \nnamespace: %s\nstatus: % s \nstart_time: %s\nfinish_time: %s\n" % (workflow.name,info_json.get('pipeline_name',''),info_json.get('describe',''),workflow.namespace,workflow.status,start_time,finish_time)
        message+='\n'
        link={
            __("pod详情"):help_url
        }
        if message:
            push_message(receivers, message, link)


# 保存workflow记录
# @pysnooper.snoop()
def save_workflow(crd, dbsession):
    pipeline_id = crd['labels'].get('pipeline-id', '')
    pipeline = dbsession.query(Pipeline).filter_by(id=int(pipeline_id)).first()
    if not pipeline:
        return None

    run_id = crd['labels'].get('run-id', '')
    alert_status = ''
    if pipeline and pipeline.alert_status:
        alert_status = pipeline.alert_status
    username = crd['username']
    print("Event: % s %s %s %s %s %s" % (crd['name'], pipeline.describe,pipeline.name,crd['username'],crd['status'],run_id))
    # print("%s status %s"%(crd['name'], crd['status']))

    # print(crd['name'],crd['namespace'])
    workflow = dbsession.query(Workflow).filter(Workflow.name == crd['name']).filter(Workflow.namespace == crd['namespace']).first()
    if workflow:
        print('exist workflow')
        workflow.status = crd['status']
        workflow.change_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        workflow.annotations = json.dumps(crd['annotations'], indent=4, ensure_ascii=False)
        workflow.labels = json.dumps(crd['labels'], indent=4, ensure_ascii=False)
        workflow.spec = json.dumps(crd['spec'], indent=4, ensure_ascii=False),
        workflow.status_more = json.dumps(crd['status_more'], indent=4, ensure_ascii=False)
        workflow.cluster = cluster
        dbsession.commit()

    else:
        info_json = {
            "pipeline_name": pipeline.name,
            "describe": pipeline.describe,
            "run_id": run_id,
            "alert_status": alert_status,
            "has_push": ''
        }
        print('new workflow')
        workflow = Workflow(name=crd['name'], cluster=cluster, namespace=crd['namespace'], create_time=crd['create_time'],
                            status=crd['status'],
                            annotations=json.dumps(crd['annotations'], indent=4, ensure_ascii=False),
                            labels=json.dumps(crd['labels'], indent=4, ensure_ascii=False),
                            spec=json.dumps(crd['spec'], indent=4, ensure_ascii=False),
                            status_more=json.dumps(crd['status_more'], indent=4, ensure_ascii=False),
                            username=username,
                            info_json=json.dumps(info_json, indent=4, ensure_ascii=False))
        dbsession.add(workflow)
        dbsession.commit()

    # 更新runhistory
    pipeline_run_id = json.loads(workflow.labels).get("run-id", '')
    if pipeline_run_id:
        run_history = dbsession.query(RunHistory).filter_by(run_id=pipeline_run_id).first()
        if run_history:
            run_history.status = crd['status']
            dbsession.commit()
    return workflow


# @pysnooper.snoop()
def check_has_push(crd, dbsession):
    workflow = dbsession.query(Workflow).filter(Workflow.name == crd['name']).filter(Workflow.namespace == crd['namespace']).first()
    if workflow and workflow.info_json:
        info_json = json.loads(workflow.info_json)
        if crd['status'] in info_json['alert_status'] and crd['status'] not in info_json['has_push']:
            return False
        else:
            return True
    return True


# 推送修改通知
# @pysnooper.snoop()
def push_resource_rec(workflow, dbsession):
    pipeline_id = json.loads(workflow.labels).get('pipeline-id', '')
    pipeline = dbsession.query(Pipeline).filter_by(id=int(pipeline_id)).first()
    if pipeline:
        init_message = __('pipeline(%s)根据近10次的任务训练资源使用情况，系统做如下调整:\n') % pipeline.describe
        message = init_message
        tasks = dbsession.query(Task).filter(Task.pipeline_id == int(pipeline_id)).all()
        for task in tasks:
            if 'NO_RESOURCE_CHECK' not in task.job_template.env.replace("-", "_").upper():
                task_monitorings = json.loads(task.monitoring).get('task', [])
                if len(task_monitorings) > 9:
                    max_cpu = 0
                    max_memory = 0
                    for task_monitoring in task_monitorings:
                        if float(task_monitoring.get('cpu', 0)) > max_cpu:
                            max_cpu = float(task_monitoring.get('cpu', 0))
                        if float(task_monitoring.get('memory', 0)) > max_memory:
                            max_memory = float(task_monitoring.get('memory', 0))
                    if max_cpu:
                        rec_cpu = math.ceil(max_cpu * 1.4) + 2
                        if rec_cpu > 150:
                            rec_cpu = 150
                        if rec_cpu != int(task.resource_cpu):
                            message += __("task(%s)，原申请cpu:%s，近10次最大使用cpu:%s，新申请值:%s\n") % (task.label, task.resource_cpu, max_cpu, rec_cpu)
                            task.resource_cpu = str(rec_cpu)
                    if max_memory:
                        rec_memory = math.ceil(max_memory * 1.4) + 2
                        if rec_memory > 350:
                            rec_memory = 350
                        if rec_memory != int(task.resource_memory.replace('G', '').replace('M', '')):
                            message += __("task(%s)，原申请mem:%s，近10次最大使用mem:%s(G)，新申请值:%s\n") % (task.label, task.resource_memory, max_memory, str(rec_memory) + "G")
                            task.resource_memory = str(rec_memory) + "G"
                    dbsession.commit()
        if message != init_message:
            alert_user = pipeline.alert_user.split(',') if pipeline.alert_user else []
            alert_user = [user.strip() for user in alert_user if user.strip()]
            receivers = alert_user + [pipeline.created_by.username]
            receivers = list(set(receivers))

            push_message(receivers, message)


# 推送训练耗时通知
# @pysnooper.snoop()
def push_task_time(workflow, dbsession):
    if not workflow:
        return

    nodes = json.loads(workflow.status_more).get('nodes', {})
    pods = {}
    for node_name in nodes:
        if nodes[node_name]['type'] == 'Pod' and nodes[node_name]['phase'] == 'Succeeded':
            pods[node_name] = nodes[node_name]
    pipeline_id = json.loads(workflow.labels).get('pipeline-id', '')
    if pipeline_id and pods:
        pipeline = dbsession.query(Pipeline).filter_by(id=pipeline_id).first()
        if pipeline:
            message = __('\n%s %s，各task耗时，酌情优化:\n') % (pipeline.describe, pipeline.created_by.username)
            task_pod_time = {}
            for pod_name in pods:
                # print(pods[pod_name])
                task_name = pods[pod_name]['displayName']
                finishedAt = datetime.datetime.strptime(pods[pod_name]['finishedAt'].replace('T',' ').replace('Z',''),'%Y-%m-%d %H:%M:%S')
                startAt = datetime.datetime.strptime(pods[pod_name]['startedAt'].replace('T', ' ').replace('Z', ''),'%Y-%m-%d %H:%M:%S')
                run_time= round((finishedAt-startAt).total_seconds()//3600,2)
                db_task_name = task_name[:task_name.index('(')] if '(' in task_name else task_name
                task = dbsession.query(Task).filter(Task.pipeline_id == int(pipeline_id)).filter(Task.name == db_task_name).first()
                if startAt in task_pod_time and task_pod_time[startAt]:
                    task_pod_time[startAt].append(
                        {
                            "task": task.label,
                            "run_time": str(run_time)
                        }
                    )
                else:
                    task_pod_time[startAt] = [
                        {
                            "task": task.label,
                            "run_time": str(run_time)
                        }
                    ]

            task_pod_time_sorted = sorted(task_pod_time.items(), key=lambda item: item[0])
            max_task_run_time = 0
            for task_pods in task_pod_time_sorted:
                for task_pod in task_pods[1]:
                    message += task_pod['task'] + ":" + task_pod['run_time'] + "(h)\n"
                    try:
                        if float(task_pod['run_time']) > max_task_run_time:
                            max_task_run_time = float(task_pod['run_time'])
                    except Exception as e:
                        print(e)

            # 记录是否已经推送，不然反复推送不好
            info_json = json.loads(workflow.info_json)
            if info_json.get('push_task_time', ''):
                pass
            else:
                info_json['push_task_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                workflow.info_json = json.dumps(info_json, indent=4, ensure_ascii=False)
                dbsession.commit()
                message += "\n"
                link = {
                    "点击查看资源的使用": "http://%s/pipeline_modelview/api/web/monitoring/%s" % (conf.get('HOST'), pipeline_id)
                }
                # 有单任务运行时长超过4个小时才通知
                if max_task_run_time > 4:
                    push_message(conf.get('ADMIN_USER').split(','), message, link)

                alert_user = pipeline.alert_user.split(',') if pipeline.alert_user else []
                alert_user = [user.strip() for user in alert_user if user.strip()]
                receivers = alert_user + [workflow.username]
                receivers = list(set(receivers))

                push_message(receivers, message, link)


# @pysnooper.snoop()
def save_monitoring(workflow, dbsession):
    try:
        if workflow.status == 'Succeeded':
            # 获取下面的所有pod
            nodes = json.loads(workflow.status_more).get('nodes', {})
            pods = {}
            for node_name in nodes:
                if nodes[node_name]['type'] == 'Pod' and nodes[node_name]['phase'] == 'Succeeded':
                    pods[node_name] = nodes[node_name]
            pipeline_id = json.loads(workflow.labels).get('pipeline-id', '')
            if pipeline_id and pods:
                for pod_name in pods:
                    print(pods[pod_name])
                    task_name = pods[pod_name]['displayName']
                    task_name = task_name[:task_name.index('(')] if '(' in task_name else task_name

                    task = dbsession.query(Task).filter(Task.pipeline_id == int(pipeline_id)).filter(Task.name == task_name).first()
                    metrics = prometheus.get_pod_resource_metric(pod_name, namespace='pipeline')
                    monitoring = json.loads(task.monitoring) if task and task.monitoring else {}
                    task_monitoring = monitoring.get('task', [])
                    if metrics:
                        task_monitoring.append({
                            "cpu": metrics.get('cpu', ''),
                            "memory": metrics.get('memory', ''),
                            "pod_name": pod_name,
                            "update_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

                    # 清理监控记录
                    task_monitoring_new = []
                    for metric in task_monitoring:
                        # 采集结果不对的，和采集结果太久远的都清理掉
                        if float(metric.get('cpu',0))>0.1 and float(metric.get('memory',0))>0.1 and metric['update_time']>(datetime.datetime.now()-datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S'):
                            task_monitoring_new.append(metric)

                    if len(task_monitoring_new) > 10:
                        del task_monitoring_new[0]

                    monitoring_new = {}
                    monitoring_new['task'] = task_monitoring_new
                    monitoring_new['tfjob'] = monitoring.get('tfjob', [])

                    print(monitoring_new)
                    if task:
                        task.monitoring = json.dumps(monitoring_new, ensure_ascii=False, indent=4)
                        dbsession.commit()

            push_task_time(workflow, dbsession)

            push_resource_rec(workflow, dbsession)

    except Exception as e:
        print(e)


# @pysnooper.snoop()
def save_history(workflow, dbsession):
    info_json = json.loads(workflow.info_json)
    if info_json['has_push']:
        if not workflow.status in info_json['has_push']:
            info_json['has_push'] += ',' + workflow.status
    else:
        info_json['has_push'] = workflow.status
    workflow.info_json = json.dumps(info_json, indent=4, ensure_ascii=False)
    dbsession.commit()


# @pysnooper.snoop()
def check_crd_exist(group, version, namespace, plural, name):
    exist_crd = client.CustomObjectsApi().get_namespaced_custom_object(group, version, namespace, plural, name)
    return exist_crd


# @pysnooper.snoop()
def deal_event(event, workflow_info, namespace):
    with session_scope(nullpool=True) as dbsession:
        try:
            crd_object = event['object']
            exist_crd = check_crd_exist(group=workflow_info['group'], version=workflow_info["version"], namespace=namespace,plural=workflow_info["plural"], name=crd_object['metadata']['name'])
            if not exist_crd:
                print('not exist in k8s')
                return
            creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
            creat_time = (datetime.datetime.strptime(creat_time,'%Y-%m-%d %H:%M:%S')+datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
            # 不能直接使用里面的状态
            status = ''
            if 'status' in crd_object and 'nodes' in crd_object['status']:
                keys = list(crd_object['status']['nodes'].keys())
                status = crd_object['status']['nodes'][keys[-1]]['phase']
                if status != 'Pending':
                    status = crd_object['status']['phase']

            back_object = {
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
            elif 'pipeline-rtx' in back_object:
                back_object['username'] = back_object['labels']['pipeline-rtx']
            workflow = save_workflow(back_object, dbsession)
            if workflow:
                has_push = check_has_push(back_object, dbsession)
                if not has_push:
                    try:
                        deliver_message(workflow, dbsession)
                    except Exception as e1:
                        print('push fail:', e1)
                        push_message(conf.get('ADMIN_USER').split(','), 'push fail' + str(e1))
                save_history(workflow, dbsession)
                save_monitoring(workflow, dbsession)

        except Exception as e:
            print(e)


# @pysnooper.snoop()
def listen_workflow():
    workflow_info = conf.get('CRD_INFO')['workflow']
    namespace = conf.get('PIPELINE_NAMESPACE')  # 不仅这一个命名空间
    w = watch.Watch()
    while (True):
        try:
            print('begin listen')
            for event in w.stream(client.CustomObjectsApi().list_namespaced_custom_object, group=workflow_info['group'],
                                  version=workflow_info["version"],
                                  namespace=namespace, plural=workflow_info["plural"]):  # label_selector=label,
                if event['type'] == 'ADDED' or event['type'] == 'MODIFIED':  # ADDED  MODIFIED DELETED
                    deal_event(event, workflow_info, namespace)
                elif event['type'] == 'ERROR':
                    w = watch.Watch()
                    time.sleep(60)

        except Exception as ee:
            print(ee)


# 不能使用异步io，因为stream会阻塞
if __name__ == '__main__':
    listen_workflow()
