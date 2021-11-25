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
conf = app.config

def create_app(script_info=None):
    return app

@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db)

import pysnooper

# https://dormousehole.readthedocs.io/en/latest/cli.html
@app.cli.command('init')
@pysnooper.snoop()
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
        add_project('job-template', 'tf分布式', 'tf相关的训练，模型校验，离线预测等功能', {"index": 4})
        add_project('job-template', 'pytorch分布式', 'pytorch相关的训练，模型校验，离线预测等功能', {"index": 5})
        add_project('job-template', 'xgb分布式', 'xgb相关的训练，模型校验，离线预测等功能', {"index": 5})
        add_project('job-template', '模型服务化', '模型服务化部署相关的组件模板', {"index": 6})
        add_project('job-template', '推荐类模板', '推荐领域常用的任务模板', {"index": 7})
        add_project('job-template', '多媒体类模板', '音视频图片常用的任务模板', {"index": 8})
        add_project('job-template', '搜索类模板', '向量搜索常用的任务模板', {"index": 9})
    except Exception as e:
        print(e)




    # 初始化创建仓库镜像模板任务流
    try:

        repository = db.session.query(Repository).filter_by(name='hubsecret').first()
        if repository is None:
            try:
                repository = Repository()
                repository.name = 'hubsecret'
                repository.server='registry.docker-cn.com'
                repository.user = 'kubeflow'
                repository.password = 'kubeflow'
                repository.hubsecret = 'hubsecret'
                repository.created_by_fk=1
                repository.changed_by_fk=1
                db.session.add(repository)
                db.session.commit()
            except Exception as e:
                db.session.rollback()


        images = db.session.query(Images).filter_by(name='ubuntu:18.04').first()
        project = db.session.query(Project).filter_by(name='基础命令').filter_by(type='job-template').first()
        if images is None and project:
            try:
                images = Images()
                images.name = 'ubuntu:18.04'
                images.describe='开源ubuntu:18.04基础镜像'
                images.created_by_fk=1
                images.changed_by_fk=1
                images.project_id=project.id
                images.repository_id=repository.id
                db.session.add(images)
                db.session.commit()
            except Exception as e:
                db.session.rollback()


        job_template = db.session.query(Job_Template).filter_by(name=conf.get('CUSTOMIZE_JOB','自定义镜像')).first()
        project = db.session.query(Project).filter_by(name='基础命令').filter_by(type='job-template').first()
        if job_template is None and project:
            try:
                job_template = Job_Template()
                job_template.name = conf.get('CUSTOMIZE_JOB','自定义镜像') if conf.get('CUSTOMIZE_JOB','自定义镜像') else '自定义镜像'
                job_template.describe='使用用户自定义镜像作为运行镜像'
                job_template.created_by_fk=1
                job_template.changed_by_fk=1
                job_template.project_id=project.id
                job_template.images_id=images.id
                job_template.version='Release'
                job_template.args=json.dumps(
                    {
                        "shell": {
                            "images": {
                                "type": "str",
                                "item_type": "str",
                                "label": "要调试的镜像",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.6",
                                "placeholder": "",
                                "describe": "要调试的镜像，<a target='_blank' href='https://github.com/tencentmusic/cube-studio/tree/master/docs/example/images'>基础镜像参考<a>",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "workdir": {
                                "type": "str",
                                "item_type": "str",
                                "label": "启动目录",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "/mnt/xx",
                                "placeholder": "",
                                "describe": "启动目录",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "command": {
                                "type": "str",
                                "item_type": "str",
                                "label": "启动命令",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "sh start.sh",
                                "placeholder": "",
                                "describe": "启动命令",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    indent=4,ensure_ascii=False)
                db.session.add(job_template)
                db.session.commit()
            except Exception as e:
                db.session.rollback()



    except Exception as e:
        print(e)


@app.cli.command('init_db')
def init_db():
    SQLALCHEMY_DATABASE_URI = conf.get('SQLALCHEMY_DATABASE_URI','')
    import sqlalchemy.engine.url as url
    uri = url.make_url(SQLALCHEMY_DATABASE_URI)
    """Inits the Myapp application"""
    import pymysql
    # 创建连接
    conn = pymysql.connect(host=uri.host,port=uri.port, user=uri.username, password=uri.password, charset='utf8')
    # 创建游标
    cursor = conn.cursor()

    # 创建数据库的sql(如果数据库存在就不创建，防止异常)
    sql = "CREATE DATABASE IF NOT EXISTS kubeflow DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;"
    # 执行创建数据库的sql
    cursor.execute(sql)
    conn.commit()



