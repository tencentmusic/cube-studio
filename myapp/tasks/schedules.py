
"""Utility functions used across Myapp"""
import logging
import random

import pysnooper
import re
import croniter
import json
import subprocess
import os
import time
import datetime
from myapp.utils.py.py_k8s import K8s
from myapp.utils.celery import session_scope
from myapp.project import push_message,push_admin
from myapp.tasks.celery_app import celery_app
# Myapp framework imports
from myapp import app
from myapp.models.model_job import (
    Pipeline,
    RunHistory,
    Workflow,
    Tfjob,
    Pytorchjob,
    Xgbjob,
    Task
)
from myapp.models.model_notebook import Notebook
from myapp.models.model_serving import InferenceService
from myapp.views.view_pipeline import run_pipeline,dag_to_pipeline
from sqlalchemy import or_

class Pusherror(Exception):
    pass


conf = app.config
logging.getLogger("task.delete_workflow").setLevel(logging.INFO)


model_map = {
    "tfjobs": Tfjob,
    "workflows": Workflow,
    "pytorchjobs": Pytorchjob,
    "xgbjobs": Xgbjob
}


# @pysnooper.snoop()
def delete_old_crd(object_info):
    timeout = int(object_info.get('timeout', 60 * 60 * 24 * 3))
    clusters = conf.get('CLUSTERS',{})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster.get('KUBECONFIG',''))

        crd_objects = []
        try:
            crd_objects = k8s_client.get_crd_all_namespaces(group=object_info['group'], version=object_info['version'],
                                                            plural=object_info['plural'], pool=False)
        except Exception as e:
            print(e)
        # print('crd_objects',crd_objects)

        with session_scope(nullpool=True) as dbsession:
            for crd_object in crd_objects:
                # print(crd_object['status'],crd_object['create_time'],crd_object['finish_time'])

                # # 如果当前还在运行，上层workflow已停止，直接删除
                # if crd_object['status']=='Running':
                run_id = json.loads(crd_object['labels']).get('run-id','').strip()
                if run_id:
                    try:
                        # 如果workflow被删除了，则下面的也一并被删除
                        workflows = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).all()
                        # print(workflows)
                        for workflow in workflows:
                            if workflow.status=='Deleted':
                                crd_names = k8s_client.delete_crd(group=object_info['group'],
                                                                  version=object_info['version'],
                                                                  plural=object_info['plural'],
                                                                  namespace=crd_object['namespace'],
                                                                  name=crd_object['name'])
                                # push_message(conf.get('ADMIN_USER', '').split(','), '%s %s 因上层workflow %s 已删除，现删除' % (object_info['plural'], crd_object['name'], run_id))
                                time.sleep(10)
                                if object_info['plural'] in model_map:
                                    db_crds = dbsession.query(model_map[object_info['plural']]).filter(model_map[object_info['plural']].name.in_(crd_names)).all()
                                    for db_crd in db_crds:
                                        db_crd.status = 'Deleted'
                                    dbsession.commit()
                    except Exception as e:
                        print(e)

                try:
                    # 如果在运行，时间比较长，就推送通知
                    if crd_object['status'] == 'Running':
                        if crd_object['create_time'] < (datetime.datetime.now() - datetime.timedelta(seconds=timeout)).strftime('%Y-%m-%d %H:%M:%S'):
                            if object_info['plural']=='workflows':
                                username=''

                                label=json.loads(crd_object['labels'])
                                pipeline_id = label.get('pipeline-id','')
                                if 'run-rtx' in label:
                                    username = label['run-rtx']
                                elif 'upload-rtx' in label:
                                    username = label['upload-rtx']
                                if username:
                                    push_message([username]+conf.get('ADMIN_USER','').split(','),'%s %s %s %s 创建时间 %s， 已经运行时间过久，注意修正'%(username,object_info['plural'],crd_object['name'],pipeline_id,crd_object['create_time']))
                    else:
                        # 如果运行结束已经1天，就直接删除
                        if crd_object['finish_time'] and crd_object['finish_time'] < (datetime.datetime.now() - datetime.timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'):
                            print('delete %s.%s namespace=%s, name=%s success' % (object_info['group'], object_info['plural'], crd_object['namespace'], crd_object['name']))
                            crd_names = k8s_client.delete_crd(group=object_info['group'], version=object_info['version'],
                                                              plural=object_info['plural'], namespace=crd_object['namespace'],
                                                              name=crd_object['name'])
                            # push_message(conf.get('ADMIN_USER', '').split(','),'%s %s %s 时运行结束，现删除 ' % (object_info['plural'], crd_object['name'], crd_object['finish_time']))
                            if object_info['plural'] in model_map:
                                db_crds = dbsession.query(model_map[object_info['plural']]).filter(model_map[object_info['plural']].name.in_(crd_names)).all()
                                for db_crd in db_crds:
                                    db_crd.status = 'Deleted'
                                dbsession.commit()
                except Exception as e:
                    print(e)



# 删除过期任务
@celery_app.task(name="task.delete_workflow", bind=True)
def delete_workflow(task):
    print('begin delete task')
    for crd_name in ["workflow","tfjob",'pytorchjob','xgbjob','mpijob','vcjob','sparkjob','paddlejob','mxjob','framework']:
        crd_info = conf.get("CRD_INFO", {}).get(crd_name, {})
        if crd_info:
            try:
                delete_old_crd(crd_info)
            except Exception as e:
                print(e)

        time.sleep(10)

    # 删除deployment
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster.get('KUBECONFIG',''))
        with session_scope(nullpool=True) as dbsession:
            try:
                deployments = k8s_client.AppsV1Api.list_deployment_for_all_namespaces().items
                # namespaces = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).all()
                # 获取所有可能的命名空间
                namespaces = dbsession.query(Workflow.namespace).group_by(Workflow.namespace).all()
                namespaces = list(set([item[0] for item in namespaces]))
                namespaces = [namespace.strip() for namespace in namespaces if namespace and namespace.strip()]
                print("workflow命名空间",namespaces)
                for deploy in deployments:
                    # print(deploy)
                    namespace = deploy.metadata.namespace
                    name = deploy.metadata.name
                    run_id = deploy.metadata.labels.get('run-id', '').strip() if deploy.metadata.labels else ''
                    # print('deployment'+namespace,name,run_id)
                    # 只处理workflow命名空间，有runid的dp
                    if namespace and namespace in namespaces and name and run_id:
                        try:
                            workflows = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).all()
                            for workflow in workflows:
                                if workflow.status == 'Succeeded' or workflow.status == 'Deleted' or workflow.status == 'Failed':
                                    print(f'delete deployment:{namespace},{name}')
                                    k8s_client.delete_deployment(namespace=namespace, name=name)
                        except Exception as e1:
                            print(e1)
            except Exception as e2:
                print(e2)
            # print(deploy)




@celery_app.task(name="task.delete_notebook", bind=True)
def delete_notebook(task):
    # 删除jupyter
    print('begin delete notebook')
    object_info = conf.get("CRD_INFO", {}).get('notebook', {})
    print(object_info)
    timeout = int(object_info.get('timeout', 60 * 60 * 24 * 3))
    namespace = conf.get('NOTEBOOK_NAMESPACE')
    with session_scope(nullpool=True) as dbsession:
        # 删除vscode的pod
        try:
            alert_time = datetime.datetime.now() - datetime.timedelta(seconds=timeout) + datetime.timedelta(days=1)
            # notebooks = dbsession.query(Notebook).filter(Notebook.changed_on < alert_time).all()   # 需要删除或者需要通知续期的notebook

            # 获取过期的gpu notebook  删除
            notebooks = dbsession.query(Notebook).filter(Notebook.changed_on < alert_time).filter(Notebook.resource_gpu!='0').all()
            for notebook in notebooks:
                if notebook.changed_on < (datetime.datetime.now() - datetime.timedelta(seconds=timeout)):
                    k8s_client = K8s(notebook.project.cluster.get('KUBECONFIG',''))
                    vscode_pods = k8s_client.get_pods(namespace=namespace,pod_name=notebook.name)
                    if vscode_pods:
                        vscode_pod=vscode_pods[0]
                        # print(vscode_pod)
                        k8s_client.delete_pods(namespace=namespace, pod_name=vscode_pod['name'])
                        user = vscode_pod['labels'].get('user', '')
                        if user:
                            pass
                            push_message([user], '您的notebook %s已清理释放资源，如果需要可reset后重新使用。' % vscode_pod['name'])
                else:
                    message = '您的notebook %s即将过期，如要继续使用，请尽快续期，每次有效期3天\n' % notebook.name
                    push_message([notebook.created_by.username], message)

        except Exception as e:
            print(e)


@celery_app.task(name="task.delete_debug_docker", bind=True)
def delete_debug_docker(task):
    clusters = conf.get('CLUSTERS',{})
    # 删除完成的任务
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        notebook_namespace = conf.get('NOTEBOOK_NAMESPACE')
        pipeline_namespace = conf.get('PIPELINE_NAMESPACE')
        k8s_client = K8s(cluster.get('KUBECONFIG',''))
        k8s_client.delete_pods(namespace=notebook_namespace,status='Succeeded')
        pipeline_pods = k8s_client.get_pods(pipeline_namespace)
        for pod in pipeline_pods:
            if pod['name'][0:6]=='debug-' or pod['name'][0:4]=='run-':
                run_id = pod['labels'].get('run-id', '')
                if run_id:
                    k8s_client.delete_workflow(all_crd_info=conf.get("CRD_INFO", {}), namespace=pipeline_namespace,run_id=run_id)
                    k8s_client.delete_pods(namespace=pipeline_namespace, labels={"run-id": run_id})

    # 删除debug和test的服务
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        namespace = conf.get('SERVICE_NAMESPACE')
        k8s_client = K8s(cluster.get('KUBECONFIG',''))
        with session_scope(nullpool=True) as dbsession:
            try:
                inferenceservices = dbsession.query(InferenceService).all()
                for inferenceservic in inferenceservices:
                    try:
                        name = 'debug-'+inferenceservic.name
                        k8s_client.delete_deployment(namespace=namespace, name=name)
                        k8s_client.delete_configmap(namespace=namespace, name=name)
                        k8s_client.delete_service(namespace=namespace, name=name)
                        k8s_client.delete_istio_ingress(namespace=namespace, name=name)
                        if inferenceservic.model_status=='debug':
                            inferenceservic.model_status='offline'
                            dbsession.commit()

                        name = 'test-' + inferenceservic.name
                        k8s_client.delete_deployment(namespace=namespace, name=name)
                        k8s_client.delete_configmap(namespace=namespace, name=name)
                        k8s_client.delete_service(namespace=namespace, name=name)
                        k8s_client.delete_istio_ingress(namespace=namespace, name=name)
                        if inferenceservic.model_status == 'test':
                            inferenceservic.model_status = 'offline'
                            dbsession.commit()

                    except Exception as e1:
                        print(e1)

            except Exception as e:
                print(e)

    push_message(conf.get('ADMIN_USER', '').split(','), 'debug pod 清理完毕')

    # 删除 sqllab jupyter 容器
    print('begin delete idex')
    namespace = conf.get('NOTEBOOK_NAMESPACE')
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster.get('KUBECONFIG',''))
        pods = k8s_client.get_pods(namespace=namespace,labels={'pod-type':"jupyter"})
        for pod in pods:
            try:
                k8s_client.v1.delete_namespaced_pod(pod['name'], namespace,grace_period_seconds=0)
            except Exception as e:
                print(e)
            try:
                k8s_client.v1.delete_namespaced_service(pod['name'], namespace, grace_period_seconds=0)
            except Exception as e:
                print(e)
            try:
                object_info = conf.get("CRD_INFO", {}).get('virtualservice', {})
                k8s_client.delete_crd(group=object_info['group'], version=object_info['version'],plural=object_info['plural'], namespace=namespace,name=pod['name'])

            except Exception as e:
                print(e)

    push_message(conf.get('ADMIN_USER', '').split(','), 'idex jupter pod 清理完毕')

    # 删除调试镜像的pod 和commit pod
    namespace = conf.get('NOTEBOOK_NAMESPACE')
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster.get('KUBECONFIG',''))
        k8s_client.delete_pods(namespace=namespace,labels={'pod-type':"docker"})

    push_message(conf.get('ADMIN_USER', '').split(','), 'docker 调试构建 pod 清理完毕')


# 推送微信消息
# @pysnooper.snoop()
def deliver_message(pipeline,message=''):
    receivers = pipeline.created_by.username.split(',')
    receivers = [receiver.strip() for receiver in receivers if receiver.strip()]
    alert_users = pipeline.alert_user.split(',') if pipeline.alert_user else []
    alert_users = [alert_user.strip() for alert_user in alert_users if alert_user.strip()]
    receivers+=alert_users
    # 失败的时候将详细推送给管理员
    receivers = list(set(receivers))
    if not receivers:
        print('no receivers')
        return

    if not message:
        message = "pipeline: %s(%s) \nnamespace: %s\ncrontab: %s\ntime: %s\nstart run" % (pipeline.name,pipeline.describe, pipeline.namespace,pipeline.cron_time,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        message = "pipeline: %s(%s) \nnamespace: %s\ncrontab: %s\ntime: %s\nfail start run:\n%s" % (pipeline.name,pipeline.describe, pipeline.namespace,pipeline.cron_time,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),message)

    push_message(receivers,message)
    # push_message(conf.get('ADMIN_USER').split(','),message)


# 获取预计发送时间。控制发送频率不要太频繁
# @pysnooper.snoop()
def next_schedules(cron_time, start_at, stop_at, resolution=0):
    crons = croniter.croniter(cron_time, start_at - datetime.timedelta(seconds=1))
    previous = start_at - datetime.timedelta(days=1)

    for eta in crons.all_next(datetime.datetime):
        # Do not cross the time boundary
        if eta >= stop_at:
            break

        if eta < start_at:
            continue

        # 去除频率过高的点
        if eta - previous < datetime.timedelta(seconds=resolution):
            continue

        yield eta
        previous = eta


# 用户每次配置定时，都会记录定时配置时间。作为start_time参考
# start_time表示最近一次的定时调度配置生效，所有处理和检测的历史，都是在start_time 之后
# 平台 worker 产生任务的进程 损坏情况恢复后  任务的时间变量模板
# 只关注created之前的任务
# 用户可以多次修改定时调度与否或者定时调度周期
# 同一个pipeline手动定时两不冲突

# 产生定时任务各个时间点的任务配置
@celery_app.task(name="task.make_timerun_config", bind=True)
def make_timerun_config(task):
    print('============= begin make timerun config')
    # 先产生所有要产生的任务。可能也会产生直接的任务。
    with session_scope(nullpool=True) as dbsession:
        try:
            resolution = conf.get("PIPELINE_TASK_CRON_RESOLUTION", 0) * 60  # 设置最小发送时间间隔，15分钟

            pipelines = dbsession.query(Pipeline).filter(Pipeline.schedule_type=='crontab').all()  # 获取model记录
            for pipeline in pipelines:  # 循环发起每一个调度
                # 无效定时时间，退出
                if not pipeline.cron_time:
                    continue
                if not re.match("^[0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+",pipeline.cron_time.strip().replace('  ', ' ')):
                    continue
                if pipeline.cronjob_start_time:
                    start_at = datetime.datetime.strptime(pipeline.cronjob_start_time,'%Y-%m-%d %H:%M:%S')
                else:
                    start_at=datetime.datetime.now()

                # 缩小指定范围，认为最后一个任务记录之前是调度记录都是已经产生的
                last_run = dbsession.query(RunHistory).filter(RunHistory.pipeline_id==pipeline.id).order_by(RunHistory.id.desc()).first()
                if last_run:
                    last_execution_date = datetime.datetime.strptime(last_run.execution_date,'%Y-%m-%d %H:%M:%S')
                    if last_execution_date>start_at:
                        start_at=last_execution_date

                stop_at = datetime.datetime.now() + datetime.timedelta(seconds=300)   # 下一个调度时间点，强制5分钟调度一次。这之前的 任务，该调度的都发起或者延迟发起

                # print('begin make timerun config %s'%pipeline.name)
                # 计算start_at和stop_at之间，每一个任务的调度时间，并保障最小周期不超过设定的resolution。
                try:
                    for eta in next_schedules(pipeline.cron_time, start_at, stop_at, resolution=resolution):  #
                        # print('执行时间点', eta)
                        execution_date = eta.strftime('%Y-%m-%d %H:%M:%S')
                        cronjob_start_time = pipeline.cronjob_start_time if pipeline.cronjob_start_time else json.loads(pipeline.expand).get("cronjob_start_time",'')
                        if cronjob_start_time and execution_date>cronjob_start_time:
                            # 要检查是否重复添加记录了
                            exist_timeruns=dbsession.query(RunHistory).filter(RunHistory.pipeline_id==pipeline.id).filter(RunHistory.execution_date==execution_date).all()
                            if not exist_timeruns:
                                pipeline_file,run_id = dag_to_pipeline(pipeline=pipeline, dbsession=dbsession,workflow_label={"schedule_type":"contab"},execution_date=execution_date)  # 合成workflow
                                # print('make pipeline file %s' % pipeline_file)
                                if pipeline_file:
                                    schedule_history = RunHistory(
                                        created_on=datetime.datetime.now(),
                                        pipeline_id=pipeline.id,
                                        pipeline_argo_id='',
                                        pipeline_file=pipeline_file,
                                        version_id='',
                                        run_id=run_id,
                                        message='',
                                        status='comed',
                                        execution_date=execution_date
                                    )
                                    dbsession.add(schedule_history)
                                    dbsession.commit()
                                else:
                                    push_message(conf.get('ADMIN_USER').split(','),'pipeline %s make config fail'%pipeline.name)
                            if len(exist_timeruns)>1:
                                for i in range(1,len(exist_timeruns)):
                                    exist_timerun = exist_timeruns[i]
                                    dbsession.delete(exist_timerun)
                                    dbsession.commit()
                                push_message(conf.get('ADMIN_USER').split(','),'发现%s 任务流在 %s 时刻存在多个定时记录'%(pipeline.name,execution_date))


                    # 无论产生任务怎么样，上传都是要执行的，可能会上传之前没有上传的任务
                    # 直接触发一次，在5分钟以内的都延迟提交。
                    # upload_timerun(pipeline,stop_at)
                except Exception as e:
                    print(e)

                upload_timerun(pipeline_id=pipeline.id,stop_time=stop_at.strftime('%Y-%m-%d %H:%M:%S'))

        except Exception as e:
            print(e)



# 计算那些任务可以准备上传了
# @pysnooper.snoop()
def upload_timerun(pipeline_id,stop_time):
    print('============= begin upload timerun')

    with session_scope(nullpool=True) as dbsession:
        try:
            pipeline = dbsession.query(Pipeline).filter(Pipeline.id == int(pipeline_id)).first()
            start_time=pipeline.cronjob_start_time
            # 获取当前pipeline  还没有处理的任务，其他的不关系
            timeruns = []
            if start_time:
                timeruns = dbsession.query(RunHistory)\
                    .filter(RunHistory.pipeline_id==pipeline.id)\
                    .filter(RunHistory.execution_date>start_time) \
                    .filter(RunHistory.execution_date <= stop_time) \
                    .filter(RunHistory.status == 'comed') \
                    .order_by(RunHistory.execution_date.desc()).all()

            if timeruns:
                # 如果依赖过去运行历史的运行状态，只检测最早的一个timerun是否可以运行
                if pipeline.depends_on_past:
                    timerun=timeruns[-1]   # 最早的一个应该调度的

                    kwargs = {
                        "timerun_id": timerun.id,
                        "pipeline_id": pipeline_id
                    }
                    # 获取前一个定时调度的timerun
                    pass_run = dbsession.query(RunHistory).filter(RunHistory.pipeline_id==pipeline.id).filter(RunHistory.execution_date>start_time).filter(RunHistory.execution_date<timerun.execution_date).order_by(RunHistory.execution_date.desc()).first()
                    if not pass_run:
                        pass
                        upload_workflow.apply_async(kwargs=kwargs,expires=120,retry=False)
                    elif pass_run.status=='created':
                        # 这里要注意处理一下 watch组件坏了，或者argo controller组件坏了的情况。以及误操作在workflow界面把记录删除了的情况
                        workflow = dbsession.query(Workflow).filter(Workflow.labels.contains(pass_run.run_id)).first()
                        if workflow:
                            if workflow.status == 'Deleted' or workflow.status == 'Succeeded':
                                print('pass workflow success finish')
                                upload_workflow.apply_async(kwargs=kwargs,expires=120,retry=False)

                        else:
                            # 直接查询实际是否有个，记录是啥，
                            crds = pipeline.get_workflow()
                            for crd in crds:
                                if pass_run.run_id in crd['labels']:
                                    # 这里可以手动把记录加进去
                                    workflow = Workflow(name=crd['name'], namespace=crd['namespace'],
                                                        create_time=crd['create_time'],
                                                        status=crd['status'],
                                                        annotations=crd['annotations'],
                                                        labels=crd['labels'],
                                                        spec=crd['spec'],
                                                        status_more=crd['status_more'],
                                                        username=pipeline.created_by.username
                                                        )
                                    dbsession.add(workflow)
                                    dbsession.commit()

                                    label = json.loads(crd['labels'])
                                    if crd['status']=='Succeeded' and label.get('run-id','')==pass_run.run_id:
                                        print('pass workflow success finish')
                                        upload_workflow.apply_async(kwargs=kwargs,expires=120,retry=False)
                # 按时间倒序，只保留最新的n个实例，之前的要删掉
                elif pipeline.expired_limit:
                    # 获取最新的n个
                    timeruns = dbsession.query(RunHistory) \
                        .filter(RunHistory.pipeline_id == pipeline.id) \
                        .filter(RunHistory.execution_date > start_time) \
                        .filter(RunHistory.execution_date <= stop_time) \
                        .order_by(RunHistory.execution_date.desc()).limit(pipeline.expired_limit)

                    latest_run_ids = [timerun.run_id for timerun in timeruns]  # 可以运行的timerun

                    # 如果有旧的在运行，就先删掉
                    exist_workflows = pipeline.get_workflow()
                    for exist_workflow in exist_workflows:
                        run_id = json.loads(exist_workflow['labels']).get('run-id', '')
                        if run_id:
                            pass_run = dbsession.query(RunHistory).filter(RunHistory.pipeline_id == pipeline.id).filter(RunHistory.execution_date > start_time).filter(RunHistory.run_id == run_id).first()
                            # 如果是定时任务发起的实例，并且已经过期，就直接删除
                            if pass_run and run_id not in latest_run_ids:
                                k8s_client = K8s(pipeline.project.cluster.get('KUBECONFIG',''))
                                k8s_client.delete_workflow(all_crd_info=conf.get("CRD_INFO", {}), namespace='pipeline',run_id=run_id)
                                workflow = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).first()
                                workflow.status='Deleted'
                                dbsession.commit()

                    # 如果有新的还没运行的，就运行
                    for timerun in timeruns:
                        if timerun.status=='comed':
                            kwargs = {
                                "timerun_id": timerun.id,
                                "pipeline_id": pipeline_id
                            }
                            upload_workflow.apply_async(kwargs=kwargs, expires=120, retry=False)



                # 按时间顺序并发运行
                else:
                    # 检测正在运行的workflow与激活并发限制是否符合
                    running_workflows = pipeline.get_workflow()
                    running_workflows = [running_workflow for running_workflow in running_workflows if running_workflow['status'] == 'Running' or running_workflow['status'] == 'Created' or running_workflow['status'] == 'Pending']
                    if len(running_workflows) < pipeline.max_active_runs:
                        more_run_num = pipeline.max_active_runs-len(running_workflows)
                        for i in range(more_run_num):
                            if len(timeruns)>i:
                                timerun=timeruns[-i-1]

                                kwargs = {
                                    "timerun_id": timerun.id,
                                    "pipeline_id": pipeline_id
                                }
                                # 执行时间会有0~5分钟差异
                                upload_workflow.apply_async(kwargs=kwargs,expires=120,retry=False)

        except Exception as e:
            print(e)




# 真正去做上传动作。
@celery_app.task(name="task.upload_workflow", bind=True)
def upload_workflow(task,timerun_id,pipeline_id):
    print('begin run workflow',timerun_id,pipeline_id)
    with session_scope(nullpool=True) as dbsession:
        try:
            pipeline = dbsession.query(Pipeline).filter(Pipeline.id == int(pipeline_id)).first()
            timerun = dbsession.query(RunHistory).filter(RunHistory.id == int(timerun_id)).first()
            # 如果想回填，可以把这个手动配置为comed
            if timerun.status=='created':
                print('timerun %s has upload'%timerun_id)
                push_message(conf.get('ADMIN_USER').split(','),'阻止重复提交 timerun %s, pipeline %s, exec time %s' % (timerun.id,pipeline.name,timerun.execution_date))
                return

            print('begin upload workflow %s %s' % (pipeline.name, datetime.datetime.now()))
            # print('read pipeline file %s' % timerun.pipeline_file)
            # return
            print('begin upload and run pipeline %s' % pipeline.name)
            try:
                json.loads(timerun.pipeline_file)
            except Exception as e:
                # 不是json要重新生成
                timerun.pipeline_file, timerun.run_id = dag_to_pipeline(pipeline=pipeline, dbsession=dbsession,
                                                        workflow_label={"schedule_type": "contab"},
                                                        execution_date=timerun.execution_date)  # 合成workflow
                dbsession.commit()
                # print(e)
            crd_name = run_pipeline(cluster=pipeline.project.cluster,workflow_json=json.loads(timerun.pipeline_file))
            timerun.pipeline_argo_id = crd_name  # pipeline_argo_id用来存储workflow的name
            timerun.status='created'

            dbsession.commit()  # 更新
            deliver_message(pipeline)   # 没有操作事务

        except Exception as e:
            print('kubeflow cronjob run pipeline error:',e)
            try:
                deliver_message(pipeline,'kubeflow cronjob run pipeline error:'+str(e))
            except Exception as e2:
                print(e2)



def delDir(dir, iteration=False):
    datatime01 = datetime.datetime.strftime(datetime.datetime.now() - datetime.timedelta(days=10), "%Y-%m-%d %H:%M:%S")
    # 获取文件夹下所有文件和文件夹
    files = os.listdir(dir)
    for file in files:
        # filepath = os.path.join(dir , file)#路径拼接
        filePath = dir + "/" + file
        # 判断是否是文件
        if os.path.isfile(filePath):
            # 最后一次修改的时间
            last1 = os.stat(filePath).st_mtime  # 获取文件的时间戳
            filetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last1))  # 将时间戳格式化成时间格式的字符串
            # 删除30天前的文件
            if (datatime01 > filetime):  # datatime01是当前时间7天前的时间，filetime是文件修改的时间，如果文件时间小于(早于)datatime01时间，就删除
                print(filePath + " was removed!", filetime)
                os.remove(filePath)

        elif os.path.isdir(filePath):
            if iteration:
                # 如果是文件夹，继续遍历删除
                delDir(filePath, iteration)
                # 如果是空文件夹，删除空文件夹
                if not os.listdir(filePath):
                    os.rmdir(filePath)
                    print(filePath + " was removed!")

# 删除过期垃圾数据
@celery_app.task(name="task.delete_old_data", bind=True)
def delete_old_data(task):
    # 获取路径
    paths = conf.get('DELETE_OLD_DATA', [])
    for path in paths:
        try:
            print('delete dir', path)
            if os.path.exists(path):
                delDir(path, iteration=True)
                print('delete dir finish', path)
                time.sleep(10)
        except Exception as e:
            print(e)

# 获取训练时长
# @pysnooper.snoop()
def get_run_time(workflow):
    start_time = json.loads(workflow.status_more).get('startedAt','')
    finish_time = json.loads(workflow.status_more).get('finishedAt', '')
    try:
        start_time = datetime.datetime.strptime(start_time.replace('T',' ').replace('Z',''),'%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(e)
        start_time=datetime.datetime.now()

    try:
        finish_time = datetime.datetime.strptime(finish_time.replace('T',' ').replace('Z',''),'%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(e)
        finish_time=datetime.datetime.now()

    return round((finish_time-start_time).total_seconds()//3600,2)

# 检查pipeline的运行时长
# @pysnooper.snoop()
def check_pipeline_time():

    with session_scope(nullpool=True) as dbsession:
        try:
            monitoring_workflow = {
            }
            today_workflows = dbsession.query(Workflow).filter(
                or_(Workflow.status == 'Running', Workflow.status == 'Succeeded')).filter(
                Workflow.create_time > datetime.datetime.now().strftime('%Y-%m-%d')).all()  # 获取model记录
            for today_workflow in today_workflows:
                # 读取
                pipeline_id = json.loads(today_workflow.labels).get('pipeline-id', '')
                if pipeline_id and pipeline_id not in monitoring_workflow:
                    pipeline = dbsession.query(Pipeline).filter(Pipeline.id == int(pipeline_id)).first()  # 获取model记录
                    monitoring_workflow[pipeline_id] = {
                        "time": [],
                        "status": today_workflow.status,
                        "user": today_workflow.username,
                        "pipeline": pipeline.describe if pipeline else '未知'
                    }
                    old_workflows = dbsession.query(Workflow).filter(Workflow.labels.contains('"pipeline-id": "%s"' % pipeline_id)).order_by(Workflow.id.desc()).limit(10).all()  # 获取model记录
                    for old_workflow in old_workflows:
                        run_time = get_run_time(old_workflow)
                        # print(old_workflow.name)
                        monitoring_workflow[pipeline_id]['time'].append(run_time)
            message = ''
            for pipeline_id in monitoring_workflow:
                work = monitoring_workflow[pipeline_id]
                message += "\npipeline:%s" % work['pipeline'] + "\nuser:%s" % work['user'] + "\nstatus:%s" % work['status'] + "\n每次训练耗时(h):%s" % work['time'] + "\n"

            print(message)
            # if message:
            #     push_admin(message)

        except Exception as e:
            print(e)


# 检查pipeline的运行资源
# @pysnooper.snoop()
def check_pipeline_resource():

    with session_scope(nullpool=True) as dbsession:
        try:
            monitoring_workflow = {}
            today_workflows = dbsession.query(Workflow).filter(Workflow.status == 'Succeeded').filter(Workflow.create_time > datetime.datetime.now().strftime('%Y-%m-%d')).all()  # 获取model记录

            for today_workflow in today_workflows:
                # 读取
                pipeline_id = json.loads(today_workflow.labels).get('pipeline-id', '')
                if pipeline_id and pipeline_id not in monitoring_workflow:
                    pipeline = dbsession.query(Pipeline).filter(Pipeline.id == int(pipeline_id)).first()  # 获取model记录
                    monitoring_workflow[pipeline_id]={
                        "user": today_workflow.username,
                        "pipeline": pipeline.describe if pipeline else '未知',
                        "task":{}
                    }
                    tasks = dbsession.query(Task).filter(Task.pipeline_id == int(pipeline_id)).all()  # 获取model记录
                    for task in tasks:
                        try:
                            task_resources= json.loads(task.monitoring).get('task',[])
                            tfjob_resources = json.loads(task.monitoring).get('tfjob',[])
                            monitoring_workflow[pipeline_id]['task'][task.label]={}
                            if task_resources:
                                monitoring_workflow[pipeline_id]['task'][task.label].update(
                                    {
                                        'cpu': [task_resource['cpu'] for task_resource in task_resources],
                                        'memory': [task_resource['memory'] for task_resource in task_resources],
                                        "cpu限制": task.resource_cpu,
                                        "memory限制" : task.resource_memory
                                    }
                                )
                            if tfjob_resources:
                                monitoring_workflow[pipeline_id]['task'][task.label].update(
                                    {
                                        "tfjob_cpu": [tfjob_resource['cpu'] for tfjob_resource in tfjob_resources],
                                        "tfjob_memory": [tfjob_resource['memory'] for tfjob_resource in tfjob_resources],
                                        "tfjob_cpu限制": re.findall('"cpu":.*', task.args)[0].replace('"cpu":','').replace('"','').replace(",",'').replace(' ',''),
                                        "tfjob_memory限制": re.findall('"memory":.*', task.args)[0].replace('"memory":','').replace('"','').replace(",",'').replace(' ','')
                                    }
                                )
                        except Exception as e:
                            print(e)


            for pipeline_id in monitoring_workflow:
                message = ''
                work = monitoring_workflow[pipeline_id]
                import copy
                work1 = copy.deepcopy(work)
                for key in work1['task']:
                    if not work1['task'][key]:
                        del work['task'][key]

                if work['task']:
                    message += "\npipeline: %s" % work['pipeline'] + "\nuser:%s" % work['user']
                    for task_name in work['task']:
                        message += "\ntask: "+task_name + "，tfjob资源使用率:"
                        message += "\n使用cpu: " + str(work['task'][task_name]['tfjob_cpu'])
                        message += "\n使用mem: " + str(work['task'][task_name]['tfjob_memory'])
                        message += "\n限制cpu: " + str(work['task'][task_name]['tfjob_cpu限制'])
                        message += "\n限制mem: " + str(work['task'][task_name]['tfjob_memory限制'])
                        message+='\n\n自行增加tfjob资源配置或worker数目'
                    print(message)
                    if message:
                        # push_message(conf.get('ADMIN_USER','').split(','),message)
                        push_message(conf.get('ADMIN_USER').split(','),message)
                        push_message([work['user']],message)

        except Exception as e:
            print(e)



@celery_app.task(name="task.check_pipeline_run", bind=True)
def check_pipeline_run(task):
    check_pipeline_time()
    check_pipeline_resource()


# @pysnooper.snoop()
def get_dir_size(dir):
    dir_size = {}
    try:
        if os.path.isdir(dir):
            command = 'ls -lh %s'%dir
            result = subprocess.getoutput(command)
            # print(result)
            rows = result.split('\n')
            for row in rows:
                row =[item for item in row.split(' ') if item]
                # print(row)
                if len(row)==9:
                    size,file_name = row[4],row[8]
                    # print(size,username)

                    if 'K' in size:
                        size = float(size.replace('K', ''))
                    elif 'M' in size:
                        size = float(size.replace('M', '')) * 1024
                    elif 'G' in size:
                        size = float(size.replace('G', '')) * 1024 * 1024
                    elif 'T' in size:
                        size = float(size.replace('T', '')) * 1024 * 1024 * 1024

                    dir_size[file_name] = round(float(size) / 1024 / 1024, 2)
                    # dir_size[file_name] = float(size) / 1024 / 1024

            # size = subprocess.check_output(command)
            # print(size)
    except Exception as e:
        print(e)

    print(dir_size)
    return dir_size



@celery_app.task(name="task.push_workspace_size", bind=True)
def push_workspace_size(task):
    # 获取路径
    paths = conf.get('CHECK_WORKSPACE_SIZE',[])

    for path in paths:
        message = '\n目录%s,目录大小前10名:\n'%path[path.rindex("/")+1:]
        print('get size dir', path)
        dir_sizes = get_dir_size(path)
        dir_sizes = sorted(dir_sizes.items(),key=lambda item:item[1],reverse=True)
        for i in range(min(10,len(dir_sizes))):
            dir_size = dir_sizes[i]
            message+=str(dir_size[0])+":"+str(dir_size[1])+"G\n"

        # push_admin(message)

        for dir_size in dir_sizes:
            user = dir_size[0]
            size = float(dir_size[1])
            if size>2500:   # 如果操作1200G，就提醒
                try:
                    push_message([user],'%s 检测到您的工作目录当前占用磁盘大小为%sG。目前每个用户工作目录上限为2500G，超出后部分功能可能受限，请及时进入个人notebook清理旧数据'%(user,str(size)))
                    push_admin('%s 检测到您的工作目录当前占用磁盘大小为%sG。目前每个用户工作目录上限为2500G，超出后部分功能可能受限，请及时进入个人notebook清理旧数据' % (user,str(size)))

                except Exception as e:
                    print(e)


@celery_app.task(name="task.watch_gpu", bind=True)
def watch_gpu(task):
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        try:
            cluster = clusters[cluster_name]
            k8s_client = K8s(cluster.get('KUBECONFIG',''))

            all_gpu_pods = k8s_client.get_uesd_gpu(namespaces=['pipeline','automl','jupyter','service'])

            print(all_gpu_pods)
            message = ''
            used_gpu = 0
            for pod in all_gpu_pods:
                used_gpu+=pod['gpu']
                message+=pod['namespace']+","+pod['user']+","+pod['name']+","+str(pod['gpu'])+"\n"
            print(message)
            message+="%s集群共已使用%s张卡"%(cluster_name,int(used_gpu))
            push_message(conf.get('ADMIN_USER','').split(','),message)

        except Exception as e1:
            print(e1)

@celery_app.task(name="task.watch_pod_utilization", bind=True)
def watch_pod_utilization(task=None):
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        try:
            cluster = clusters[cluster_name]
            k8s_client = K8s(cluster.get('KUBECONFIG', ''))

            # 获取连续2天的低利用率pod要报警
            from myapp.utils.py.py_prometheus import Prometheus
            prometheus = Prometheus(conf.get('PROMETHEUS', 'prometheus-k8s.monitoring:9090'))

            service_pods = k8s_client.get_pods(namespace='service')
            service_pods_metrics = prometheus.get_namespace_resource_metric(namespace="service")
            for pod in service_pods:
                if pod['start_time'] > (datetime.datetime.now() - datetime.timedelta(days=2)) and pod['name'] in service_pods_metrics and pod['username']:
                    try:
                        if pod['cpu'] > 5 and service_pods_metrics[pod['name']]['cpu'] < pod['cpu'] / 5:
                            push_message([pod['username']] + conf.get('ADMIN_USER', '').split(','),f'集群 {cluster_name} 用户 {pod["username"]} pod {pod["name"]}资源cpu使用率过低，最新2天最大使用率为{round(service_pods_metrics[pod["name"]]["cpu"],2)}，但申请值为{pod["cpu"]}，请及时清理或修改申请值')

                        # 虚拟gpu服务不考虑
                        if int(pod.get('gpu', 0)) >= 1 and service_pods_metrics[pod['name']]['gpu'] < 0.15:
                            push_message([pod['username']] + conf.get('ADMIN_USER', '').split(','),f'集群 {cluster_name} 用户 {pod["username"]} pod {pod["name"]}资源gpu使用率过低，最新2天最大使用率为{round(service_pods_metrics[pod["name"]]["gpu"],2)}，但申请值为{pod["gpu"]}，请及时清理或修改申请值')
                            pass
                    except Exception as e:
                        print(e)

        except Exception as e1:
            print(e1)

        # push_admin("%s集群共已使用%s张卡"%(cluster_name,int(used_gpu)))


# 各项目组之间相互均衡的方案，一台机器上可能并不能被一个项目组占完，所以可能会跑多个项目组的任务
@celery_app.task(name="task.adjust_node_resource", bind=True)
def adjust_node_resource(task):
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster.get('KUBECONFIG',''))
        all_node = k8s_client.get_node()
        all_node_json = {}
        pending_pods={}
        # 获取每台机器的资源容纳量
        for node in all_node:  # list 转dict
            ip = node['hostip']
            if node['labels'].get('share','true')=='true' and node['labels'].get('train','false')=='true':  # 前提要求机器允许被其他项目组共享
                if node['labels'].get('cpu','false')=='true' or node['labels'].get('gpu','false')=='true':
                    all_node_json[ip] = node
                    all_node_json[ip]['used_memory'] = []
                    all_node_json[ip]['used_cpu'] = []
                    all_node_json[ip]['used_gpu'] = []

        # print(all_node_json)
        for namespace in ['jupyter', 'pipeline', 'automl', 'service']:
            all_pods = k8s_client.get_pods(namespace=namespace)
            for pod in all_pods:
                if pod['host_ip'] not in all_node_json:
                    continue
                if pod['status'] == 'Running':
                    # print(namespace,pod)
                    all_node_json[pod['host_ip']]['used_memory'].append(pod['memory'])
                    all_node_json[pod['host_ip']]['used_cpu'].append(pod['cpu'])
                    all_node_json[pod['host_ip']]['used_gpu'].append(pod['gpu'])
                    # print(all_node_json[pod['host_ip']])
                # 有挂起等待超过5分钟的情况，立刻划资源过去，并推送通知，因为挂起不一定是因为资源。
                if pod['status']=='Pending' and (datetime.datetime.now()-pod['start_time']).total_seconds()>300:
                    # 如果因为资源不足就通过资源调度解决
                    containers = pod['status_more'].get('conditions', [])
                    messages = ','.join([container['message'] if container['message'] else '' for container in containers])

                    if 'insufficient' in messages.lower():
                        pending_pods[pod['name']]={
                            "namespace":namespace,
                            "cluster":cluster_name,
                            "node_selector":pod['node_selector']
                        }
                        push_message(conf.get('ADMIN_USER','').split(','),'cluster %s, namespace %s pod %s 因资源问题 pending'%(cluster_name,namespace,pod['name']))
                    else:
                        push_message(conf.get('ADMIN_USER', '').split(','),'cluster %s, namespace %s pod %s 因其他问题 pending' % (cluster_name,namespace, pod['name']))

        for ip in all_node_json:
            all_node_json[ip]['used_memory'] = int(sum(all_node_json[ip]['used_memory']))
            all_node_json[ip]['used_cpu'] = int(sum(all_node_json[ip]['used_cpu']))
            all_node_json[ip]['used_gpu'] = int(sum(all_node_json[ip]['used_gpu']))


        # 获取每个资源组的资源申请量，cpu机器和gpu单独看。
        all_org_resource={}
        for ip in all_node_json:
            org=all_node_json[ip]['labels'].get('org','public')
            if org not in all_org_resource:
                all_org_resource[org]={
                    "cpu_node_num":0,
                    "gpu_node_num":0,
                    "cpu_req_total":0,
                    "gpu_req_total": 0,
                    "cpu_allocatable_total":0,
                    "gpu_allocatable_total":0
                }
            if all_node_json[ip]['labels'].get('cpu','false')=='true':
                all_org_resource[org]['cpu_node_num']+=1
                all_org_resource[org]['cpu_req_total'] += all_node_json[ip]['used_cpu']
                all_org_resource[org]['cpu_allocatable_total'] += all_node_json[ip]['cpu']

            if all_node_json[ip]['labels'].get('gpu','false')=='true':
                all_org_resource[org]['gpu_node_num']+=1
                all_org_resource[org]['gpu_req_total'] += all_node_json[ip]['used_gpu']
                all_org_resource[org]['gpu_allocatable_total'] += all_node_json[ip]['gpu']

        # 计算申请率最大最小集群
        max_cpu_org=max_gpu_org=min_cpu_org=min_gpu_org='public'
        max_cpu_per = max_gpu_per = 0
        min_cpu_per = min_gpu_per = 1
        for org in all_org_resource:
            org_resource=all_org_resource[org]
            if org_resource['cpu_node_num']>2:   # 至少3台机器，才参与调度融合
                if org_resource['cpu_req_total']/org_resource['cpu_allocatable_total']>max_cpu_per:
                    max_cpu_per=org_resource['cpu_req_total']/org_resource['cpu_allocatable_total']
                    max_cpu_org=org
                if org_resource['cpu_req_total']/org_resource['cpu_allocatable_total']<min_cpu_per:
                    min_cpu_per=org_resource['cpu_req_total']/org_resource['cpu_allocatable_total']
                    min_cpu_org=org

            if org_resource['gpu_node_num']>2:   # 至少3台机器，才参与调度融合
                if org_resource['gpu_req_total']/org_resource['gpu_allocatable_total']>max_gpu_per:
                    max_gpu_per=org_resource['gpu_req_total']/org_resource['gpu_allocatable_total']
                    max_gpu_org=org
                if org_resource['gpu_req_total']/org_resource['gpu_allocatable_total']<min_gpu_per:
                    min_gpu_per=org_resource['gpu_req_total']/org_resource['gpu_allocatable_total']
                    min_gpu_org=org

        # 获取项目组下面，每台机器的cpu申请量
        def get_cpu_per_node(org):
            org_node_cpu_per = {}
            for ip in all_node_json:
                if all_node_json[ip]['labels'].get('org', '') == org and all_node_json[ip]['labels'].get('cpu','false') == 'true':
                    org_node_cpu_per[ip] = all_node_json[ip]['used_cpu'] / all_node_json[ip]['cpu']

            org_node_cpu_per = sorted(org_node_cpu_per.items(), key=lambda x: x[1], reverse=False)  # 从小到大排序
            return org_node_cpu_per

        # 获取项目组下面，每台机器的gpu申请量
        def get_gpu_per_node(org):
            org_node_gpu_per={}
            for ip in all_node_json:
                if all_node_json[ip]['labels'].get('org','')==org and all_node_json[ip]['labels'].get('gpu','false')=='true':
                    org_node_gpu_per[ip]=all_node_json[ip]['used_gpu']/all_node_json[ip]['gpu']
            org_node_gpu_per = sorted(org_node_gpu_per.items(), key=lambda x: x[1], reverse=False)   # 从小到大排序
            return org_node_gpu_per


        # 如果存在资源问题pending，直接调整
        if pending_pods:
            for pod_name in pending_pods:
                des_org = pending_pods[pod_name]['node_selector'].get('org','public')
                # 如果缺少cpu
                if pending_pods[pod_name]['node_selector'].get('cpu','false')=='true' and des_org!=min_cpu_org:
                    # 直接将申请量最小的集群中申请量最小的cpu机器迁移过去
                    org_node_cpu_per = get_cpu_per_node(min_cpu_org)
                    print(org_node_cpu_per)
                    adjust_node = [node[0] for node in org_node_cpu_per[:1]]  # 每次调整一台机器
                    push_message(conf.get('ADMIN_USER').split(','), '集群 %s 调整项目组 %s 下 cpu机器 %s 到项目组%s' % (cluster_name, min_cpu_org, ','.join(adjust_node), des_org))
                    k8s_client.label_node(adjust_node, labels={"org": des_org})
                    return

                if pending_pods[pod_name]['node_selector'].get('gpu','false')=='true' and des_org!=min_gpu_org:
                    org_node_gpu_per = get_gpu_per_node(min_gpu_org)
                    print(org_node_gpu_per)
                    adjust_node = [node[0] for node in org_node_gpu_per[:1]]  # 每次调整一台机器
                    push_message(conf.get('ADMIN_USER').split(','), '集群 %s 调整项目组 %s 下 gpu机器 %s 到项目组%s' % (cluster_name, min_gpu_org, ','.join(adjust_node), des_org))
                    k8s_client.label_node(adjust_node, labels={"org": des_org})
                    return

        # 不存在资源挂起的情况，保持最大最小集群申请量差异在20%以下
        print(all_org_resource)
        # 如果差别最大的两个不同的资源组，cpu申请率差距在20%，则将申请率最小的资源组中的申请率最小的机器转为到另一个资源组
        print(max_cpu_org,min_cpu_org,max_gpu_org,min_gpu_org)
        if max_cpu_org!=min_cpu_org and max_cpu_per>min_cpu_per+0.2:
            org_node_cpu_per = get_cpu_per_node(min_cpu_org)
            print(org_node_cpu_per)
            adjust_node = [node[0] for node in org_node_cpu_per[:1]]   # 每次调整一台机器
            push_message(conf.get('ADMIN_USER').split(','),'集群 %s 调整项目组 %s 下 cpu机器 %s 到项目组%s'%(cluster_name,min_cpu_org,','.join(adjust_node),max_cpu_org))
            k8s_client.label_node(adjust_node,labels={"org":max_cpu_org})
            return


        # 将差距最大的两个gpu资源组，进行调配
        if max_gpu_org!=min_gpu_org and max_gpu_per>min_gpu_per+0.2:
            org_node_gpu_per = get_gpu_per_node(min_gpu_org)
            print(org_node_gpu_per)
            adjust_node = [node[0] for node in org_node_gpu_per[:1]]  # 每次调整一台机器
            push_message(conf.get('ADMIN_USER').split(','), '集群 %s 调整项目组 %s 下 gpu机器 %s 到项目组%s' % (cluster_name, min_gpu_org, ','.join(adjust_node), max_gpu_org))
            k8s_client.label_node(adjust_node,labels={"org":max_gpu_org})
            return


# get_dir_size('/data/k8s/kubeflow/pipeline/workspace')
# @pysnooper.snoop()
def get_deployment_node_selector(name,namespace):
    from kubernetes import client
    exist_dp = client.AppsV1Api().read_namespaced_deployment(name=name, namespace=namespace)

    node_selector = {}
    try:
        # aa=client.V1NodeSelector
        if exist_dp.spec.template.spec.affinity.node_affinity and exist_dp.spec.template.spec.affinity.node_affinity.required_during_scheduling_ignored_during_execution:
            match_expressions = exist_dp.spec.template.spec.affinity.node_affinity.required_during_scheduling_ignored_during_execution.node_selector_terms
            match_expressions = [ex.match_expressions for ex in match_expressions]
            match_expressions = match_expressions[0]
            for match_expression in match_expressions:
                if match_expression.operator == 'In':
                    node_selector[match_expression.key] = match_expression.values[0]
                if match_expression.operator == 'Equal':
                    node_selector[match_expression.key] = match_expression.values

    except Exception as e:
        print(e)
        pass

        # print(e)
    if exist_dp.spec.template.spec.node_selector:
        node_selector.update(exist_dp.spec.template.spec.node_selector)

    print(node_selector)

    pass


# 不同优先级的服务之间调节算力
@celery_app.task(name="task.adjust_service_resource", bind=True)
def adjust_service_resource(task):
    from kubernetes import client
    cluster_name='tke'
    namespace = conf.get('SERVICE_NAMESPACE')
    cluster = conf.get('CLUSTERS', {})[cluster_name]
    with session_scope(nullpool=True) as dbsession:
        try:
            k8s_client = K8s(cluster.get('KUBECONFIG',''))
            hpas = client.AutoscalingV2beta1Api().list_namespaced_horizontal_pod_autoscaler(namespace=namespace).items
            for hpa in hpas:
                inferenceserving = dbsession.query(InferenceService).filter_by(name=hpa.metadata.name).filter_by(model_status='online').first()
                if not inferenceserving:
                    message = cluster_name + "：请删除hpa，因" + hpa.metadata.name + '服务下线或者不存在'
                    push_message(conf.get('ADMIN_USER').split(','), message=message)
                    continue
                else:
                    if inferenceserving.resource_gpu and inferenceserving.resource_gpu!='0' and inferenceserving.priority==1:
                        current_replicas = hpa.status.current_replicas
                        desired_replicas = hpa.status.desired_replicas
                        if desired_replicas>current_replicas:  # 期望扩容
                            pass
                            # 如果没有扩张，或者持续时间太久，就缩小低优先级服务
                            if not hpa.status.last_scale_time or datetime.datetime.now().timestamp() - hpa.status.last_scale_time.astimezone(datetime.timezone(datetime.timedelta(hours=8))).timestamp() > 400:
                                push_message(conf.get('ADMIN_USER').split(','),'寻找扩服务%s一卡'%(inferenceserving.name,))
                                target_node_selector = get_deployment_node_selector(name=inferenceserving.name,namespace=namespace)

                                # 获取同项目组，低优先级的推理
                                low_inferenceservings =dbsession.query(InferenceService).filter_by(priority=0).filter_by(project_id=inferenceserving.project_id).all()
                                low_inferenceservings.sort(key=lambda item:item.max_replicas-item.min_replicas)  # 从大到小排序
                                for service in low_inferenceservings:
                                    if service.resource_gpu and service.resource_gpu!='0':  #
                                        current_replicas = client.AppsV1Api().read_namespaced_deployment(name=service.name, namespace=namespace).spec.replicas
                                        # 如果当前副本数大于最小副本数
                                        if current_replicas > service.min_replicas:
                                            # 随意缩放一个pod
                                            if not target_node_selector.get('gpu-type',''):
                                                client.AppsV1Api().patch_namespaced_deployment_scale(service.name, namespace,[{'op': 'replace', 'path': '/spec/replicas', 'value': current_replicas-1}])
                                                push_message([service.created_by.username,inferenceserving.created_by.username]+conf.get('ADMIN_USER').split(','),'缩服务%s一卡，扩服务%s一卡'%(service.name,inferenceserving.name))
                                                return
                                            # 缩放指定pod
                                            else:
                                                node_selector = get_deployment_node_selector(name=service.name,namespace=namespace)
                                                target_gpu_type = target_node_selector['gpu-type']
                                                exist_gpu_type = node_selector.get('gpu-type','')
                                                if exist_gpu_type and exist_gpu_type!=target_gpu_type:
                                                    print('服务gpu卡型不匹配')
                                                    break
                                                # 如果低级别服务没有gpu机型限制。就查看是否有符合需求的机器型号，缩放指定pod
                                                pods = k8s_client.get_pods(namespace=namespace,labels={"app":service.name,"pod-type":"inference"})
                                                nodeips = [pod['host_ip'] for pod in pods]
                                                for nodeip in nodeips:
                                                    node = k8s_client.get_node(nodeip)
                                                    if node['labels'].get('gpu-type','')==target_gpu_type:
                                                        # 缩放指定pod
                                                        can_scale_pods = [pod for pod in pods if pod['host_ip']==nodeip]
                                                        if can_scale_pods:
                                                            k8s_client.v1.delete_namespaced_pod(can_scale_pods[0]['name'], namespace,grace_period_seconds=0)
                                                            client.AppsV1Api().patch_namespaced_deployment_scale(service.name, namespace, [{'op': 'replace', 'path': '/spec/replicas','value': current_replicas - 1}])
                                                            push_message([service.created_by.username,inferenceserving.created_by.username] + conf.get('ADMIN_USER').split(','), '缩服务%s一卡，扩服务%s一卡' % (service.name, inferenceserving.name))

                                                            return


        except Exception as e:
            print(e)


from myapp.models.model_aihub import Aihub
def add_aihub(info_path):
    if not os.path.exists(info_path):
        return
    aihubs = json.load(open(info_path, mode='r'))
    with session_scope(nullpool=True) as dbsession:
        try:
            if len(aihubs)>0:
                # dbsession.query(Aihub).delete()
                # dbsession.commit()
                for data in aihubs:
                    print(data)
                    name = data.get('name','')
                    label = data.get('label','')
                    describe = data.get('describe','')
                    uuid = data.get('uuid', '')
                    if name and label and describe and uuid:
                        aihub = dbsession.query(Aihub).filter_by(uuid=uuid).first()
                        if not aihub:
                            aihub=Aihub()
                        aihub.doc=data.get('doc','')
                        aihub.name=name
                        aihub.label=label
                        aihub.describe=describe
                        aihub.field=data.get('field','')
                        aihub.scenes=data.get('scenes','')
                        aihub.type=data.get('type','')
                        aihub.pic=data.get('pic','')
                        aihub.status=data.get('status', '')
                        aihub.uuid=uuid
                        aihub.images = data.get('images', '')
                        aihub.version=data.get('version', '')
                        aihub.dataset=json.dumps(data.get('dataset', {}),indent=4,ensure_ascii=False)
                        aihub.notebook=json.dumps(data.get('notebook', {}),indent=4,ensure_ascii=False)
                        aihub.job_template=json.dumps(data.get('train', {}),indent=4,ensure_ascii=False)
                        aihub.pre_train_model=json.dumps(data.get('pre_train_model', {}),indent=4,ensure_ascii=False)
                        aihub.inference=json.dumps(data.get('inference', {}),indent=4,ensure_ascii=False)
                        aihub.service=json.dumps(data.get('service', {}),indent=4,ensure_ascii=False)
                        aihub.hot=int(data.get('hot', '0'))
                        aihub.price=int(data.get('price', '0'))
                        aihub.source=data.get('source', '')
                        if not aihub.id:
                            dbsession.add(aihub)
                        dbsession.commit()
        except Exception as e:
            print(e)


def cp_cubestudio():
    # 复制cube-studio代码
    import os
    from myapp.utils import core
    des_path = f'/data/k8s/kubeflow/global/'
    os.makedirs(des_path, exist_ok=True)
    try:
        core.run_shell(f'cp -rf /cube-studio {des_path}')
    except Exception as e:
        print(e)

@celery_app.task(name="task.update_aihub", bind=True)
def update_aihub(task):
    import random
    time.sleep(random.randint(10,600))
    from myapp.utils.core import run_shell

    # 更新git
    info_path='info.json'
    status = run_shell('rm -rf /cube-studio && cd / && git clone https://github.com/tencentmusic/cube-studio.git')
    if status:
        print('clone fail')
        return
    else:
        if os.path.exists(info_path):
            info_path = '/cube-studio/aihub/info.json'
            cp_cubestudio()
    add_aihub(info_path)

if __name__=="__main__":
    add_aihub('info.json')



