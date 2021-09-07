
"""Utility functions used across Myapp"""
import sys,os
import numpy as np
from bs4 import BeautifulSoup
import requests,base64,hashlib
from collections import namedtuple
import datetime
from email.utils import make_msgid, parseaddr
import logging
import time,json
from urllib.error import URLError
import urllib.request
import pysnooper
import re
import croniter
from dateutil.tz import tzlocal
import shutil
import os,sys,io,json,datetime,time
import subprocess
from datetime import datetime, timedelta
import os
import sys
import time
import datetime
from myapp.utils.py.py_k8s import K8s
from myapp.utils.celery import session_scope
from myapp.project import push_message,push_admin
from myapp.tasks.celery_app import celery_app
# Myapp framework imports
from myapp import app, db, security_manager
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
from myapp.security import (
    MyUser
)
from myapp.views.view_pipeline import run_pipeline,dag_to_pipeline
from sqlalchemy.exc import InvalidRequestError,OperationalError
from sqlalchemy import or_

class Pusherror(Exception):
    pass


conf = app.config
logging.getLogger("task.delete_tfjob").setLevel(logging.INFO)


model_map = {
    "tfjobs": Tfjob,
    "workflows": Workflow,
    "pytorchjobs": Pytorchjob,
    "xgbjobs": Xgbjob
}


@pysnooper.snoop()
def delete_old_crd(object_info):
    timeout = int(object_info.get('timeout', 60 * 60 * 24 * 3))
    clusters = conf.get('CLUSTERS',{})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster['KUBECONFIG'])

        crd_objects = []
        try:
            crd_objects = k8s_client.get_crd_all_namespaces(group=object_info['group'], version=object_info['version'],
                                                            plural=object_info['plural'], pool=False)
        except Exception as e:
            print(e)
        # print('crd_objects',crd_objects)

        with session_scope(nullpool=True) as dbsession:
            for crd_object in crd_objects:
                print(crd_object['status'],crd_object['create_time'],crd_object['finish_time'])

                # # 如果当前还在运行，上层workflow已停止，直接删除
                # if crd_object['status']=='Running':
                run_id = json.loads(crd_object['labels']).get('run-id','').strip()
                if run_id:
                    try:
                        # 如果workflow被删除了，则下面的也一并被删除
                        workflows = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).all()
                        print(workflows)
                        for workflow in workflows:
                            if workflow.status=='Deleted':
                                crd_names = k8s_client.delete_crd(group=object_info['group'],
                                                                  version=object_info['version'],
                                                                  plural=object_info['plural'],
                                                                  namespace=crd_object['namespace'],
                                                                  name=crd_object['name'])
                                time.sleep(10)
                                if model_map[object_info['plural']] in model_map:
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
                                if 'run-rtx' in label:
                                    username = label['run-rtx']
                                elif 'upload-rtx' in label:
                                    username = label['upload-rtx']
                                if username:
                                    push_message([username]+conf.get('ADMIN_USER','').split(','),'%s %s 创建时间 %s， 已经运行时间过久，注意修正'%(object_info['plural'],crd_object['name'],crd_object['create_time']))
                    else:
                        # 如果运行结束已经1天，就直接删除
                        if crd_object['finish_time'] and crd_object['finish_time'] < (datetime.datetime.now() - datetime.timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'):
                            print('delete %s.%s namespace=%s, name=%s success' % (object_info['group'], object_info['plural'], crd_object['namespace'], crd_object['name']))
                            crd_names = k8s_client.delete_crd(group=object_info['group'], version=object_info['version'],
                                                              plural=object_info['plural'], namespace=crd_object['namespace'],
                                                              name=crd_object['name'])
                            if object_info['plural'] in model_map:
                                db_crds = dbsession.query(model_map[object_info['plural']]).filter(model_map[object_info['plural']].name.in_(crd_names)).all()
                                for db_crd in db_crds:
                                    db_crd.status = 'Deleted'
                                dbsession.commit()
                except Exception as e:
                    print(e)



# 删除过期任务
@celery_app.task(name="task.delete_tfjob", bind=True)
def delete_tfjob(task):
    print('begin delete task')

    workflow_info = conf.get("CRD_INFO", {}).get('workflow', {})
    print(workflow_info)
    if workflow_info:
        delete_old_crd(workflow_info)

    time.sleep(10)

    tfjob_info = conf.get("CRD_INFO", {}).get('tfjob', {})
    print(tfjob_info)
    if tfjob_info:
        delete_old_crd(tfjob_info)

    time.sleep(10)

    pytorchjob_info = conf.get("CRD_INFO", {}).get('pytorchjob', {})
    print(pytorchjob_info)
    if pytorchjob_info:
        delete_old_crd(pytorchjob_info)

    time.sleep(10)


    xgbjob_info = conf.get("CRD_INFO", {}).get('xgbjob', {})
    print(xgbjob_info)
    if xgbjob_info:
        delete_old_crd(xgbjob_info)

    time.sleep(10)

    xgbjob_info = conf.get("CRD_INFO", {}).get('mpijob', {})
    print(xgbjob_info)
    if xgbjob_info:
        delete_old_crd(xgbjob_info)

    time.sleep(10)

    # 删除framework
    framework_info = conf.get("CRD_INFO", {}).get('framework', {})
    print(framework_info)
    if framework_info:
        delete_old_crd(framework_info)

    time.sleep(10)


    # 删除deployment
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster['KUBECONFIG'])

        deployments = k8s_client.AppsV1Api.list_namespaced_deployment(namespace='pipeline').items
        for deploy in deployments:
            run_id = deploy.metadata.labels.get('run-id', '').strip()
            if run_id:
                with session_scope(nullpool=True) as dbsession:
                    try:
                        workflows = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).all()
                        for workflow in workflows:
                            if workflow.status == 'Succeeded' or workflow.status == 'Deleted' or workflow.status == 'Failed':
                                k8s_client.delete_deployment(namespace='pipeline', name=deploy.name)
                    except Exception as e:
                        print(e)

            # print(deploy)
            try:
                create_time = deploy.metadata.creation_timestamp.strftime('%Y-%m-%d')
                delete_time=(datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                if create_time < delete_time:
                    print('kill %s'%deploy.metadata.name)
                    k8s_client.delete_deployment(namespace='pipeline', name=deploy.name)
            except Exception as e:
                print(e)



    time.sleep(60)


    # 删除daemon
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        try:
            k8s_client = K8s(cluster['KUBECONFIG'])

            daemon_sets = k8s_client.AppsV1Api.list_namespaced_daemon_set(namespace='pipeline').items
            for daemon_set in daemon_sets:
                # print(deploy)
                run_id = daemon_set.metadata.labels.get('run-id', '').strip()
                if run_id:
                    with session_scope(nullpool=True) as dbsession:
                        try:
                            workflows = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).all()
                            for workflow in workflows:
                                if workflow.status == 'Succeeded' or workflow.status == 'Deleted' or workflow.status == 'Failed':
                                    k8s_client.AppsV1Api.delete_namespaced_daemon_set(namespace='pipeline', name=daemon_set.name)
                        except Exception as e:
                            print(e)

                try:
                    create_time = daemon_set.metadata.creation_timestamp.strftime('%Y-%m-%d')
                    delete_time=(datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                    if create_time < delete_time:
                        print('kill %s'%daemon_set.metadata.name)
                        k8s_client.AppsV1Api.delete_namespaced_daemon_set(namespace='pipeline', name=daemon_set.name)
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

    time.sleep(60)

    # 删除sts
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        try:
            k8s_client = K8s(cluster['KUBECONFIG'])

            stss = k8s_client.AppsV1Api.list_namespaced_stateful_set(namespace='pipeline').items
            for sts in stss:
                run_id = sts.metadata.labels.get('run-id', '').strip()
                if run_id:
                    with session_scope(nullpool=True) as dbsession:
                        try:
                            workflows = dbsession.query(Workflow).filter(Workflow.labels.contains(run_id)).all()
                            for workflow in workflows:
                                if workflow.status == 'Succeeded' or workflow.status == 'Deleted' or workflow.status == 'Failed':
                                    k8s_client.AppsV1Api.delete_namespaced_stateful_set(namespace='pipeline', name=sts.name)
                        except Exception as e:
                            print(e)
                try:
                    create_time = sts.metadata.creation_timestamp.strftime('%Y-%m-%d')
                    delete_time=(datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                    if create_time < delete_time:
                        print('kill %s'%sts.metadata.name)
                        k8s_client.AppsV1Api.delete_namespaced_stateful_set(namespace='pipeline', name=sts.name)
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

    time.sleep(60)

    # 删除pod
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        try:
            k8s_client = K8s(cluster['KUBECONFIG'])

            pods = k8s_client.v1.list_namespaced_pod(namespace='pipeline').items
            for pod in pods:
                # print(pod)
                try:
                    create_time = pod.metadata.creation_timestamp.strftime('%Y-%m-%d')
                    delete_time=(datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                    if create_time < delete_time:
                        print('kill %s'%pod.metadata.name)
                        k8s_client.v1.delete_namespaced_pod(namespace='pipeline', name=pod.metadata.name)
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

    push_message(conf.get('ADMIN_USER','').split(','),'清理历史pod完成')


@celery_app.task(name="task.delete_notebook", bind=True)
def delete_notebook(task):
    # 删除jupyter
    print('begin delete notebook')
    object_info = conf.get("CRD_INFO", {}).get('notebook', {})
    print(object_info)
    timeout = int(object_info.get('timeout', 60 * 60 * 24 * 3))

    clusters = conf.get('CLUSTERS',{})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster['KUBECONFIG'])

        with session_scope(nullpool=True) as dbsession:
            # 删除vscode的pod
            try:
                vscode_pods = k8s_client.get_pods(namespace=conf.get('NOTEBOOK_NAMESPACE'))
                for vscode_pod in vscode_pods:
                    notebook = dbsession.query(Notebook).filter_by(name=vscode_pod['name']).first()  # 获取model记录
                    if notebook:
                        if notebook.changed_on < (datetime.datetime.now() - datetime.timedelta(seconds=timeout)):
                            k8s_client.delete_pods(namespace=conf.get('NOTEBOOK_NAMESPACE'),pod_name=vscode_pod['name'])
                            user = vscode_pod['lables'].get('user','')
                            if user:
                                push_message([user],'您的notebook %s已清理释放资源，如果需要可reset后重新使用。'%vscode_pod['name'])
                    else:
                        k8s_client.delete_pods(namespace=conf.get('NOTEBOOK_NAMESPACE'), pod_name=vscode_pod['name'])
            except Exception as e:
                print(e)





@celery_app.task(name="task.alert_notebook_renew", bind=True)
def alert_notebook_renew(task):
    with session_scope(nullpool=True) as dbsession:
        # 删除vscode的pod
        try:
            object_info = conf.get("CRD_INFO", {}).get('notebook', {})
            timeout = int(object_info.get('timeout', 60 * 60 * 24 * 3))

            start = datetime.datetime.now()-datetime.timedelta(seconds=timeout)+datetime.timedelta(days=2)
            end = datetime.datetime.now()-datetime.timedelta(seconds=timeout)
            notebooks = dbsession.query(Notebook).filter(Notebook.changed_on>=end).filter(Notebook.changed_on<=start).all()
            for notebook in notebooks:
                message='您的notebook %s即将过期，如要继续使用，请尽快续期，每次有效期2天\n'%notebook.name
                push_message([notebook.created_by.username],message)
        except Exception as e:
            print(e)

    push_message(conf.get('ADMIN_USER','').split(','),'notebook续期通知完成')




# 推送微信消息
# @pysnooper.snoop()
def deliver_message(pipeline,message=''):
    receivers = pipeline.created_by.username.split(',')
    receivers = [receiver.strip() for receiver in receivers if receiver.strip()]
    alert_users = pipeline.alert_user.split(',') if pipeline.alert_user else []
    alert_users = [alert_user.strip() for alert_user in alert_users if alert_user.strip()]
    receivers+=alert_users
    # 失败的时候将详细推送给管理员
    # if message:
    #     bcc = conf.get('PIPELINE_TASK_BCC_ADDRESS','')  # 暗抄送列表
    #     bcc = bcc.split(',')
    #     for bc in bcc:
    #         receivers.append(bc)
    receivers = list(set(receivers))
    if not receivers:
        print('no receivers')
        return

    if not message:
        message = "pipeline: %s(%s) \nnamespace: %s\ncrontab: %s\ntime: %s\nstart run" % (pipeline.name,pipeline.describe, pipeline.namespace,pipeline.cron_time,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        message = "pipeline: %s(%s) \nnamespace: %s\ncrontab: %s\ntime: %s\nfail start run:\n%s" % (pipeline.name,pipeline.describe, pipeline.namespace,pipeline.cron_time,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),message)

    push_message(receivers,message)
    push_message(conf.get('ADMIN_USER').split(','),message)


# @pysnooper.snoop()
def save_history(dbsession,pipeline,message=''):
    schedule_history = RunHistory(
        created_on=datetime.datetime.now(),
        pipeline_id=pipeline.id,
        pipeline_argo_id=pipeline.pipeline_id,
        pipeline_file=pipeline.pipeline_file,
        version_id=pipeline.version_id,
        run_id=pipeline.run_id,
        message=message
    )
    dbsession.add(schedule_history)
    dbsession.commit()


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
            # Get the top of the hour
            start_at = datetime.datetime.now(tzlocal()).replace(microsecond=0, second=0, minute=0)  # 当前小时整点
            stop_at = start_at + datetime.timedelta(seconds=3600)  # 下一个小时整点

            pipelines = dbsession.query(Pipeline).filter(Pipeline.schedule_type=='crontab').all()  # 获取model记录
            for pipeline in pipelines:  # 循环发起每一个调度

                print('begin make timerun config %s'%pipeline.name)
                # 计算start_at和stop_at之间，每一个任务的调度时间，并保障最小周期不超过设定的resolution。
                for eta in next_schedules(pipeline.cron_time, start_at, stop_at, resolution=resolution):  #
                    print('执行时间点', eta)
                    execution_date = eta.strftime('%Y-%m-%d %H:%M:%S')
                    if execution_date>pipeline.cronjob_start_time:
                        # 要检查是否重复添加记录了
                        exist_timeruns=dbsession.query(RunHistory).filter(RunHistory.pipeline_id==pipeline.id).filter(RunHistory.execution_date==execution_date).first()
                        if not exist_timeruns:
                            pipeline_file = dag_to_pipeline(pipeline=pipeline, dbsession=dbsession,execution_date=execution_date)  # 合成workflow
                            # print('make pipeline file %s' % pipeline_file)
                            if pipeline_file:
                                schedule_history = RunHistory(
                                    created_on=datetime.datetime.now(),
                                    pipeline_id=pipeline.id,
                                    pipeline_argo_id='',
                                    pipeline_file=pipeline_file,
                                    version_id='',
                                    run_id='',
                                    message='',
                                    status='comed',
                                    execution_date=execution_date
                                )
                                dbsession.add(schedule_history)
                                dbsession.commit()
                            else:
                                push_message(conf.get('ADMIN_USER').split(','),'pipeline %s make config fail'%pipeline.name)

        except Exception as e:
            print(e)





# 定时执行workflow调度任务   每5分钟
@celery_app.task(name="task.upload_timerun", bind=True)
def upload_timerun(task):
    print('============= begin upload timerun ')

    with session_scope(nullpool=True) as dbsession:
        try:
            pipelines = dbsession.query(Pipeline).filter(Pipeline.schedule_type=='crontab').all()  # 获取model记录
            for pipeline in pipelines:  # 循环发起每一个调度
                start_time=pipeline.cronjob_start_time
                # 获取当前pipeline  还没有处理的任务，其他的不关系
                timeruns = dbsession.query(RunHistory)\
                    .filter(RunHistory.pipeline_id==pipeline.id)\
                    .filter(RunHistory.execution_date>start_time)\
                    .filter(RunHistory.status == 'comed') \
                    .order_by(RunHistory.execution_date.desc()).all()

                if timeruns:
                    # 如果依赖过去运行历史的运行状态，只检测最早的一个timerun是否可以运行
                    if pipeline.depends_on_past:
                        timerun=timeruns[-1]   # 最早的一个应该调度的
                        # 获取前一个定时调度的timerun
                        pass_run = dbsession.query(RunHistory).filter(RunHistory.pipeline_id==pipeline.id).filter(RunHistory.execution_date>start_time).filter(RunHistory.execution_date<timerun.execution_date).order_by(RunHistory.execution_date.desc()).first()
                        if not pass_run:
                            upload_workflow(timerun, pipeline, dbsession)
                        elif pass_run.status=='created':
                            # 这里要注意处理一下 watch组件坏了，或者argo controller组件坏了的情况。以及误操作在workflow界面把记录删除了的情况
                            workflow = dbsession.query(Workflow).filter(Workflow.labels.contains(pass_run.run_id)).first()
                            if workflow:
                                if workflow.status == 'Deleted' or workflow.status == 'Succeeded':
                                    print('pass workflow success finish')
                                    upload_workflow(timerun,pipeline,dbsession)

                    else:
                        # 检测正在运行的workflow与限制是否符合
                        running_workflows = pipeline.get_workflow()
                        running_workflows = [running_workflow for running_workflow in running_workflows if running_workflow['status'] == 'Running' or running_workflow['status'] == 'Created' or running_workflow['status'] == 'Pending']
                        if len(running_workflows) < pipeline.max_active_runs:
                            more_run_num = pipeline.max_active_runs-len(running_workflows)
                            for i in range(more_run_num):
                                if len(timeruns)>i:
                                    upload_workflow(timeruns[-i-1], pipeline, dbsession)

        except Exception as e:
            print(e)




# 依据timerun配置发起调度
# @pysnooper.snoop()
def upload_workflow(timerun,pipeline,dbsession):

    try:
        print('begin upload workflow %s %s' % (pipeline.name, datetime.datetime.now()))
        print('read pipeline file %s' % timerun.pipeline_file)
        # return
        print('begin upload and run pipeline %s' % pipeline.name)

        pipeline_argo_id,version_id,run_id = run_pipeline(
            pipeline_file=timerun.pipeline_file,
            pipeline_name=pipeline.name,
            kfp_host=pipeline.project.cluster.get('KFP_HOST'),
            pipeline_argo_id=timerun.pipeline_argo_id,
            pipeline_argo_version_id=timerun.version_id
        )
        print('success upload and run pipeline %s,pipeline_argo_id %s, version_id %s,run_id %s ' % (pipeline.name,pipeline_argo_id,version_id,run_id))
        timerun.pipeline_argo_id = pipeline_argo_id
        timerun.version_id = version_id
        timerun.run_id = run_id
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
        print('delete dir', path)
        if os.path.exists(path):
            delDir(path, iteration=True)
            print('delete dir finish', path)
            time.sleep(10)


# 获取训练时长
@pysnooper.snoop()
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

    return round((finish_time-start_time).seconds/60/60,2)

# 检查pipeline的运行时长
@pysnooper.snoop()
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
                    old_workflows = dbsession.query(Workflow).filter(
                        Workflow.labels.contains('"pipeline-id": "%s"' % pipeline_id)).order_by(Workflow.id.desc()).limit(10).all()  # 获取model记录
                    for old_workflow in old_workflows:
                        run_time = get_run_time(old_workflow)
                        # print(old_workflow.name)
                        monitoring_workflow[pipeline_id]['time'].append(run_time)
            message = ''
            for pipeline_id in monitoring_workflow:
                work = monitoring_workflow[pipeline_id]
                message += "\npipeline:%s" % work['pipeline'] + "\nuser:%s" % work['user'] + "\nstatus:%s" % work[
                    'status'] + "\n每次训练耗时(h):%s" % work['time'] + "\n"

            print(message)
            if message:
                push_admin(message)

        except Exception as e:
            print(e)


# 检查pipeline的运行资源
@pysnooper.snoop()
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
                            # if task_resources:
                            #     monitoring_workflow[pipeline_id]['task'][task.label].update(
                            #         {
                            #             'cpu': [task_resource['cpu'] for task_resource in task_resources],
                            #             'memory': [task_resource['memory'] for task_resource in task_resources],
                            #             "cpu限制": task.resource_cpu,
                            #             "memory限制" : task.resource_memory
                            #         }
                            #     )
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


# 获取目录的大小
def get_dir_size(dir):
    dir_size={}
    files = os.listdir(dir)
    for file in files:
        filePath = dir + "/" + file
        if os.path.isdir(filePath):
            """disk usage in human readable format (e.g. '2,1GB')"""
            size = subprocess.check_output(['du','-sh', filePath]).split()[0].decode('utf-8')
            print(file, size)
            if 'K' in size:
                size=float(size.replace('K',''))
            elif 'M' in size:
                size=float(size.replace('M',''))*1024
            elif 'G' in size:
                size=float(size.replace('G',''))*1024*1024
            elif 'T' in size:
                size=float(size.replace('T',''))*1024*1024*1024

            dir_size[file]=round(float(size)/1024/1024,2)

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

        push_admin(message)

        for dir_size in dir_sizes:
            user = dir_size[0]
            size = float(dir_size[1])
            if size>1200:   # 如果操作1200G，就提醒
                try:
                    push_message([user],'检测到您的工作目录当前占用磁盘大小为%sG。目前每个用户工作目录上限为1500G，超出后部分功能可能受限，请及时进入个人notebook清理旧数据'%str(size))
                except Exception as e:
                    print(e)


@celery_app.task(name="task.watch_gpu", bind=True)
def watch_gpu(task):
    clusters = conf.get('CLUSTERS', {})
    for cluster_name in clusters:
        cluster = clusters[cluster_name]
        k8s_client = K8s(cluster['KUBECONFIG'])

        all_gpu_pods=k8s_client.get_uesd_gpu(['pipeline','katib','jupyter','service'])

        print(all_gpu_pods)
        message = ''
        used_gpu = 0
        for pod in all_gpu_pods:
            used_gpu+=pod['gpu']
            message+=pod['namespace']+","+pod['user']+","+pod['name']+","+str(pod['gpu'])+"\n"
        print(message)
        message+="%s集群共已使用%s张卡"%(cluster_name,int(used_gpu))
        push_message([conf.get('ADMIN_USER','')],message)
        push_admin("%s集群共已使用%s张卡"%(cluster_name,int(used_gpu)))


# if __name__ == '__main__':
#     delete_tfjob(task=None)




