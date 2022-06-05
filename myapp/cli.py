#!/usr/bin/env python
from datetime import datetime
import logging
from subprocess import Popen
from sys import stdout

import click
from colorama import Fore, Style
from flask import g
import json
from myapp import app, appbuilder, db, security_manager
from myapp.models.model_notebook import Notebook
from myapp.models.model_team import Project,Project_User
from myapp.models.model_job import Repository,Images,Job_Template,Pipeline,Task
from myapp.models.model_serving import Service,InferenceService
conf = app.config
import requests

def create_app(script_info=None):
    return app

@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db)

import pysnooper

# https://dormousehole.readthedocs.io/en/latest/cli.html
@app.cli.command('init')
# @pysnooper.snoop()
def init():

    # 初始化创建项目组
    try:
        """Inits the Myapp application"""
        appbuilder.add_permissions(update_perms=True)   # update_perms为true才会检测新权限
        security_manager.sync_role_definitions()
        def add_project(project_type,name,describe,expand={}):
            project = db.session.query(Project).filter_by(name=name).filter_by(type=project_type).first()
            if project is None:
                try:
                    project = Project()
                    project.type=project_type
                    project.name = name
                    project.describe=describe
                    project.expand=json.dumps(expand,ensure_ascii=False,indent=4)
                    db.session.add(project)
                    db.session.commit()

                    project_user = Project_User()
                    project_user.project=project
                    project_user.role = 'creator'
                    project_user.user_id=1
                    db.session.add(project_user)
                    db.session.commit()

                except Exception as e:
                    db.session.rollback()


        # 添加一些默认的记录
        add_project('org','推荐中心','推荐项目组')
        add_project('org', '多媒体中心', '多媒体项目组')
        add_project('org', '搜索中心', '搜索项目组')
        add_project('org', '广告中心', '广告项目组')
        add_project('org', 'public', '公共项目组')

        add_project('job-template', '基础命令', 'python/bash等直接在服务器命令行中执行命令的模板',{"index":1})
        add_project('job-template', '数据导入导出', '集群与用户机器或其他集群之间的数据迁移',{"index":2})
        add_project('job-template', '数据处理', '数据的单机或分布式处理任务',{"index":3})
        add_project('job-template', '机器学习', '传统机器学习，lr/决策树/gbdt/xgb/fm等', {"index": 4})
        add_project('job-template', 'tf分布式', 'tf相关的训练，模型校验，离线预测等功能', {"index": 5})
        add_project('job-template', 'pytorch分布式', 'pytorch相关的训练，模型校验，离线预测等功能', {"index": 6})
        add_project('job-template', 'xgb分布式', 'xgb相关的训练，模型校验，离线预测等功能', {"index": 7})
        add_project('job-template', '模型服务化', '模型服务化部署相关的组件模板', {"index": 8})
        add_project('job-template', '推荐类模板', '推荐领域常用的任务模板', {"index": 9})
        add_project('job-template', '多媒体类模板', '音视频图片文本常用的任务模板', {"index": 10})
        add_project('job-template', '搜索类模板', '向量搜索常用的任务模板', {"index": 11})
    except Exception as e:
        print(e)


    def create_template(repository_id,project_name,image_name,image_describe,job_template_name,job_template_describe='',job_template_command='',job_template_args=None,job_template_volume='',job_template_account='',job_template_expand=None,job_template_env=''):
        if not repository_id:
            return
        images = db.session.query(Images).filter_by(name=image_name).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='job-template').first()
        if images is None and project:
            try:
                images = Images()
                images.name = image_name
                images.describe=image_describe
                images.created_by_fk=1
                images.changed_by_fk=1
                images.project_id=project.id
                images.repository_id=repository_id
                db.session.add(images)
                db.session.commit()
            except Exception as e:
                db.session.rollback()


        job_template = db.session.query(Job_Template).filter_by(name=job_template_name).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='job-template').first()
        if project and images.id:
            if job_template is None:
                try:
                    job_template = Job_Template()
                    job_template.name = job_template_name.replace('_','-')
                    job_template.describe=job_template_describe
                    job_template.entrypoint=job_template_command,
                    job_template.volume_mount=job_template_volume,
                    job_template.accounts=job_template_account,
                    job_template.expand = json.dumps(job_template_expand,indent=4,ensure_ascii=False) if job_template_expand else '{}'
                    job_template.created_by_fk=1
                    job_template.changed_by_fk=1
                    job_template.project_id=project.id
                    job_template.images_id=images.id
                    job_template.version='Release'
                    job_template.env=job_template_env
                    job_template.args=json.dumps(job_template_args,indent=4,ensure_ascii=False) if job_template_args else '{}'
                    db.session.add(job_template)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
            else:
                try:
                    job_template.describe = job_template_describe
                    job_template.entrypoint = job_template_command,
                    job_template.volume_mount = job_template_volume,
                    job_template.accounts = job_template_account,
                    job_template.expand = json.dumps(job_template_expand, indent=4,ensure_ascii=False) if job_template_expand else '{}'
                    job_template.created_by_fk = 1
                    job_template.changed_by_fk = 1
                    job_template.project_id = project.id
                    job_template.images_id = images.id
                    job_template.version = 'Release'
                    job_template.env = job_template_env
                    job_template.args = json.dumps(job_template_args, indent=4,ensure_ascii=False) if job_template_args else '{}'
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()




    # 初始化创建仓库镜像模板任务流
    try:
        repository = db.session.query(Repository).filter_by(name='hubsecret').first()
        if repository is None:
            try:
                repository = Repository()
                repository.name = 'hubsecret'
                repository.server='registry.docker-cn.com'
                repository.user = 'yourname'
                repository.password = 'yourpassword'
                repository.hubsecret = 'hubsecret'
                repository.created_by_fk=1
                repository.changed_by_fk=1
                db.session.add(repository)
                db.session.commit()
            except Exception as e:
                db.session.rollback()

        job_templates = json.load(open('myapp/init-job-template.json',mode='r'))
        for job_template_name in job_templates:
            job_template = job_templates[job_template_name]
            job_template['repository_id']=repository.id
            create_template(**job_template)

    except Exception as e:
        print(e)



    # 添加demo 服务
    def create_service(project_name,service_name,service_describe,image_name,command,env,resource_mem='2G',resource_cpu='2',ports='80'):
        service = db.session.query(Service).filter_by(name=service_name).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='org').first()
        if service is None and project:
            try:
                service = Service()
                service.name = service_name.replace('_','-')
                service.label=service_describe
                service.created_by_fk=1
                service.changed_by_fk=1
                service.project_id=project.id
                service.images=image_name
                service.command = command
                service.env='\n'.join([x.strip() for x in env.split('\n') if x.split()])
                service.ports = ports
                db.session.add(service)
                db.session.commit()
            except Exception as e:
                db.session.rollback()

    try:

        services = json.load(open('myapp/init-service.json',mode='r'))
        for service_name in services:
            service = services[service_name]
            create_service(**service)
    except Exception as e:
        print(e)




    # 创建demo pipeline
    def create_pipeline(tasks,pipeline):
        # 如果项目组或者task的模板不存在就丢失
        org_project = db.session.query(Project).filter_by(name=pipeline['project']).filter_by(type='org').first()
        if not org_project:
            return
        for task in tasks:
            job_template = db.session.query(Job_Template).filter_by(name=task['job_templete']).first()
            if not job_template:
                return


        # 创建pipeline
        pipeline_model = db.session.query(Pipeline).filter_by(name=pipeline['name']).first()
        if pipeline_model is None:
            try:
                pipeline_model = Pipeline()
                pipeline_model.name = pipeline['name']
                pipeline_model.describe = pipeline['describe']
                pipeline_model.dag_json=json.dumps(pipeline['dag_json'],indent=4,ensure_ascii=False).replace('_','-')
                pipeline_model.created_by_fk = 1
                pipeline_model.changed_by_fk = 1
                pipeline_model.project_id = org_project.id
                pipeline_model.parameter = json.dumps(pipeline.get('parameter',{}))
                db.session.add(pipeline_model)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
        else:
            pipeline_model.describe = pipeline['describe']
            pipeline_model.dag_json = json.dumps(pipeline['dag_json'],indent=4,ensure_ascii=False).replace('_', '-')
            pipeline_model.created_by_fk = 1
            pipeline_model.changed_by_fk = 1
            pipeline_model.project_id = org_project.id
            pipeline_model.parameter = json.dumps(pipeline.get('parameter', {}))
            db.session.commit()


        # 创建task
        for task in tasks:
            task_model = db.session.query(Task).filter_by(name=task['name']).filter_by(pipeline_id=pipeline_model.id).first()
            job_template = db.session.query(Job_Template).filter_by(name=task['job_templete']).first()
            if task_model is None and job_template:
                try:
                    task_model = Task()
                    task_model.name = task['name'].replace('_','-')
                    task_model.label = task['label']
                    task_model.args = json.dumps(task['args'],indent=4,ensure_ascii=False)
                    task_model.volume_mount = task.get('volume_mount','')
                    task_model.resource_memory = task.get('resource_memory','2G')
                    task_model.resource_cpu = task.get('resource_cpu','2')
                    task_model.resource_gpu = task.get('resource_gpu','0')
                    task_model.created_by_fk = 1
                    task_model.changed_by_fk = 1
                    task_model.pipeline_id = pipeline_model.id
                    task_model.job_template_id = job_template.id
                    db.session.add(task_model)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
            else:
                task_model.label = task['label']
                task_model.args = json.dumps(task['args'],indent=4,ensure_ascii=False)
                task_model.volume_mount = task.get('volume_mount', '')
                task_model.node_selector = task.get('node_selector', 'cpu=true,train=true,org=public')
                task_model.retry = int(task.get('retry', 0))
                task_model.timeout = int(task.get('timeout', 0))
                task_model.resource_memory = task.get('resource_memory', '2G')
                task_model.resource_cpu = task.get('resource_cpu', '2')
                task_model.resource_gpu = task.get('resource_gpu', '0')
                task_model.created_by_fk = 1
                task_model.changed_by_fk = 1
                task_model.pipeline_id = pipeline_model.id
                task_model.job_template_id = job_template.id
                db.session.commit()

        # 修正pipeline
        pipeline_model.dag_json = pipeline_model.fix_dag_json()  # 修正 dag_json
        pipeline_model.expand = json.dumps(pipeline_model.fix_expand(), indent=4, ensure_ascii=False)   # 修正 前端expand字段缺失
        pipeline_model.expand = json.dumps(pipeline_model.fix_position(), indent=4, ensure_ascii=False)  # 修正 节点中心位置到视图中间
        db.session.commit()
        # 自动排版
        db_tasks = pipeline_model.get_tasks(db.session)
        if db_tasks:
            try:
                tasks={}
                for task in db_tasks:
                    tasks[task.name]=task.to_json()

                from myapp.utils import core
                expand = core.fix_task_position(pipeline_model.to_json(),tasks,json.loads(pipeline_model.expand))
                pipeline_model.expand=json.dumps(expand,indent=4,ensure_ascii=False)
                db.session.commit()
            except Exception as e:
                print(e)


    try:
        pipelines = json.load(open('myapp/init-pipeline.json',mode='r'))
        for pipeline_name in pipelines:
            pipeline = pipelines[pipeline_name]['pipeline']
            tasks = pipelines[pipeline_name]['tasks']
            create_pipeline(pipeline=pipeline,tasks=tasks)
    except Exception as e:
        print(e)


