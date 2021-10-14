
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
from myapp.models.model_docker import Docker
conf = app.config



@celery_app.task(name="task.check_docker_commit", bind=True)  # , soft_time_limit=15
@pysnooper.snoop()
def check_docker_commit(task,docker_id):  # 在页面中测试时会自定接收者和id
    with session_scope(nullpool=True) as dbsession:
        try:
            docker = dbsession.query(Docker).filter_by(id=int(docker_id)).first()
            pod_name = "docker-commit-%s-%s" % (docker.created_by.username, str(docker.id))
            namespace = conf.get('NOTEBOOK_NAMESPACE')
            k8s_client = K8s(conf.get('CLUSTERS').get(conf.get('ENVIRONMENT')).get('KUBECONFIG'))
            begin_time=datetime.datetime.now()
            now_time=datetime.datetime.now()
            while((now_time-begin_time).seconds<1800):   # 也就是最多commit push 30分钟
                time.sleep(12000)
                commit_pods = k8s_client.get_pods(namespace=namespace,pod_name=pod_name)
                if commit_pods:
                    commit_pod=commit_pods[0]
                    if commit_pod['status']=='Succeeded':
                        docker.last_image=docker.target_image
                        dbsession.commit()
                        break
                    # 其他异常状态直接报警
                    if commit_pod['status']!='Running':
                        push_message(conf.get('ADMIN_USER').split(','),'commit pod %s not running'%commit_pod['name'])
                        break
                else:
                    break

        except Exception as e:
            print(e)



