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
conf = app.config

def create_app(script_info=None):
    return app

@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db)

import pysnooper

# https://dormousehole.readthedocs.io/en/latest/cli.html
@app.cli.command('init')
def init():
    try:
        """Inits the Myapp application"""
        appbuilder.add_permissions(update_perms=True)   # update_perms为true才会检测新权限
        security_manager.sync_role_definitions()
        def add_project(type,name,describe,expand={}):
            project = db.session.query(Project).filter_by(name=name).first()
            if project is None:
                try:
                    project = Project()
                    project.type=type
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



