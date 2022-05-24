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
        add_project('job-template', '机器学习', '传统机器学习，lr/决策树/gbdt/xgb/fm等', {"index": 4})
        add_project('job-template', 'tf分布式', 'tf相关的训练，模型校验，离线预测等功能', {"index": 5})
        add_project('job-template', 'pytorch分布式', 'pytorch相关的训练，模型校验，离线预测等功能', {"index": 6})
        add_project('job-template', 'xgb分布式', 'xgb相关的训练，模型校验，离线预测等功能', {"index": 7})
        add_project('job-template', '模型服务化', '模型服务化部署相关的组件模板', {"index": 8})
        add_project('job-template', '推荐类模板', '推荐领域常用的任务模板', {"index": 9})
        add_project('job-template', '多媒体类模板', '音视频图片常用的任务模板', {"index": 10})
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
        if job_template is None and project and images.id:
            try:
                job_template = Job_Template()
                job_template.name = job_template_name
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

        # 注册自定义镜像
        create_template(
            repository_id=repository.id,
            project_name='基础命令',
            image_name='ubuntu:18.04',
            image_describe='开源ubuntu:18.04基础镜像',
            job_template_name=conf.get('CUSTOMIZE_JOB','自定义镜像') if conf.get('CUSTOMIZE_JOB','自定义镜像') else '自定义镜像',
            job_template_describe='使用用户自定义镜像作为运行镜像',
            job_template_args={
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
                        "describe": "要调试的镜像，<a target='_blank' href='https://github.com/tencentmusic/cube-studio/tree/master/imagess'>基础镜像参考<a>",
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
            }
        )





        # datax数据导入导出
        create_template(
            repository_id=repository.id,
            project_name='数据导入导出',
            image_name='ai.tencentmusic.com/tme-public/datax:latest',
            image_describe='datax异构数据源同步',
            job_template_name='datax',
            job_template_describe='datax异构数据源同步',
            job_template_command='',
            job_template_volume='',
            job_template_account='',
            job_template_env='',
            job_template_expand={
                "index": 1,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/datax"
            },

            job_template_args={
                "shell": {
                    "-f": {
                        "type": "str",
                        "item_type": "str",
                        "label": "job.json文件地址",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "/usr/local/datax/job/job.json",
                        "placeholder": "",
                        "describe": "job.json文件地址",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )

        # 注册volcano模板
        create_template(
            repository_id=repository.id,
            project_name='数据处理',
            image_name='ai.tencentmusic.com/tme-public/volcano:20211001',
            image_describe='有序分布式任务',
            job_template_name='volcanojob',
            job_template_describe='有序分布式任务',
            job_template_command='',
            job_template_volume='kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_expand={
                "index": 1,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/volcano"
            },
            job_template_env='''NO_RESOURCE_CHECK=true
            TASK_RESOURCE_CPU=2
            TASK_RESOURCE_MEMORY=4G
            TASK_RESOURCE_GPU=0
            '''.strip().replace(' ',''),
            job_template_args={
                "shell": {
                    "--working_dir": {
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
                    "--command": {
                        "type": "str",
                        "item_type": "str",
                        "label": "启动命令",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "echo aa",
                        "placeholder": "",
                        "describe": "启动命令",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--num_worker": {
                        "type": "str",
                        "item_type": "str",
                        "label": "占用机器个数",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "3",
                        "placeholder": "",
                        "describe": "占用机器个数",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--image": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.6",
                        "placeholder": "",
                        "describe": "worker镜像，直接运行你代码的环境镜像<a href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }

        )



        # 注册ray分布式
        create_template(
            repository_id=repository.id,
            project_name='数据处理',
            image_name='ai.tencentmusic.com/tme-public/ray:gpu-20210601',
            image_describe='ray分布式任务',
            job_template_name='ray',
            job_template_describe='python多机分布式任务，数据处理',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm',
            job_template_account='kubeflow-pipeline',
            job_template_expand={
                "index": 2,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/ray"
            },
            job_template_args={
                "shell": {
                    "-n": {
                        "type": "int",
                        "item_type": "",
                        "label": "分布式任务worker的数量",
                        "require": 1,
                        "choice": [],
                        "range": "$min,$max",
                        "default": "3",
                        "placeholder": "",
                        "describe": "分布式任务worker的数量",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "-i": {
                        "type": "str",
                        "item_type": "str",
                        "label": "每个worker的初始化脚本文件地址，用来安装环境",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "每个worker的初始化脚本文件地址，用来安装环境",
                        "describe": "每个worker的初始化脚本文件地址，用来安装环境",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "-f": {
                        "type": "str",
                        "item_type": "str",
                        "label": "python启动命令，例如 python3 /mnt/xx/xx.py",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "python启动命令，例如 python3 /mnt/xx/xx.py",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )




        # sklear分布式
        create_template(
            repository_id=repository.id,
            project_name='机器学习',
            image_name='ai.tencentmusic.com/tme-public/sklearn_estimator:v1',
            image_describe='sklearn基于ray的分布式',
            job_template_name='ray-sklearn',
            job_template_describe='sklearn基于ray的分布式',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
                NO_RESOURCE_CHECK=true
                ''',
            job_template_expand={
                "index": 8,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/ray_sklearn"
            },

            job_template_args={
                "shell": {
                    "--train_csv_file_path": {
                        "type": "str",
                        "item_type": "str",
                        "label": "训练集csv",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "训练集csv，|分割符，首行是列名",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--predict_csv_file_path": {
                        "type": "str",
                        "item_type": "str",
                        "label": "预测数据集csv",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "预测数据集csv，格式和训练集一致，默认为空，需要predict时填",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--label_name": {
                        "type": "str",
                        "item_type": "str",
                        "label": "label的列名，必填",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "label的列名，必填",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--model_name": {
                        "type": "str",
                        "item_type": "str",
                        "label": "模型名称，必填",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "训练用到的模型名称，如LogisticRegression，必填。常用的都支持，要加联系管理员",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--model_args_dict": {
                        "type": "str",
                        "item_type": "str",
                        "label": "模型参数",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "模型参数，json格式，默认为空",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--model_file_path": {
                        "type": "str",
                        "item_type": "str",
                        "label": "模型文件保存文件名，必填",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "模型文件保存文件名，必填",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--predict_result_path": {
                        "type": "str",
                        "item_type": "str",
                        "label": "预测结果保存文件名，默认为空，需要predict时填",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "预测结果保存文件名，默认为空，需要predict时填",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--worker_num": {
                        "type": "str",
                        "item_type": "str",
                        "label": "ray worker数量",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "ray worker数量",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )




        # 注册tf runner分布式
        create_template(
            repository_id=repository.id,
            project_name='tf分布式',
            image_name='ai.tencentmusic.com/tme-public/tf2.3_keras_train:latest',
            image_describe='tf分布式-runner方式',
            job_template_name='tfjob-runner',
            job_template_describe='tf分布式-runner方式',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
            NO_RESOURCE_CHECK=true
            TASK_RESOURCE_CPU=4
            TASK_RESOURCE_MEMORY=4G
            TASK_RESOURCE_GPU=0
            '''.strip().replace(' ',''),
            job_template_expand={
                "index": 1,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/tf_keras_train"
            },

            job_template_args={
                "shell": {
                    "--job": {
                        "type": "json",
                        "item_type": "str",
                        "label": "模型训练，json配置",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "模型训练，json配置",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--upstream-output-file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "上游输出文件",
                        "require": 0,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "上游输出文件",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )




        # 注册tf plain分布式
        create_template(
            repository_id=repository.id,
            project_name='tf分布式',
            image_name='ai.tencentmusic.com/tme-public/tf2.3_plain_train:latest',
            image_describe='tf分布式-plain方式',
            job_template_name='tfjob-plain',
            job_template_describe='tf分布式-plain方式',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
            NO_RESOURCE_CHECK=true
            TASK_RESOURCE_CPU=4
            TASK_RESOURCE_MEMORY=4G
            TASK_RESOURCE_GPU=0
            '''.strip().replace(' ',''),
            job_template_expand={
                "index": 2,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/tf_plain_train"
            },

            job_template_args={
                "shell": {
                    "--job": {
                        "type": "json",
                        "item_type": "str",
                        "label": "模型训练，json配置",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "模型训练，json配置",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--upstream-output-file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "上游输出文件",
                        "require": 0,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "上游输出文件",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )




        # 注册tf 分布式训练
        create_template(
            repository_id=repository.id,
            project_name='tf分布式',
            image_name='ai.tencentmusic.com/tme-public/tf_distributed_train:latest',
            image_describe='tf分布式训练',
            job_template_name='tfjob-train',
            job_template_describe='tf分布式训练，内部支持plain和runner两种方式',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
            NO_RESOURCE_CHECK=true
            TASK_RESOURCE_CPU=4
            TASK_RESOURCE_MEMORY=4G
            TASK_RESOURCE_GPU=0
            '''.strip().replace(' ',''),
            job_template_expand={
                "index": 3,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/tf_distributed_train"
            },

            job_template_args={
                "shell": {
                    "--job": {
                        "type": "json",
                        "item_type": "str",
                        "label": "模型训练，json配置",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "模型训练，json配置",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--upstream-output-file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "上游输出文件",
                        "require": 0,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "上游输出文件",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )

        # 注册tf 模型评估
        create_template(
            repository_id=repository.id,
            project_name='tf分布式',
            image_name='ai.tencentmusic.com/tme-public/tf2.3_model_evaluation:latest',
            image_describe='tensorflow2.3模型评估',
            job_template_name='tf-model-evaluation',
            job_template_describe='tensorflow2.3模型评估',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
             NO_RESOURCE_CHECK=true
             TASK_RESOURCE_CPU=4
             TASK_RESOURCE_MEMORY=4G
             TASK_RESOURCE_GPU=0
             '''.strip().replace(' ', ''),
            job_template_expand={
                "index": 4,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/tf_model_evaluation"
            },

            job_template_args={
                "shell": {
                    "--job": {
                        "type": "json",
                        "item_type": "str",
                        "label": "模型对比评估，json配置",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "模型对比评估，json配置",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--upstream-output-file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "上游输出文件",
                        "require": 0,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "上游输出文件",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )

        # 注册tf 模型分布式评估
        create_template(
            repository_id=repository.id,
            project_name='tf分布式',
            image_name='ai.tencentmusic.com/tme-public/tf_distributed_eval:latest',
            image_describe='tensorflow2.3分布式模型评估',
            job_template_name='tf-distribute-model-evaluation',
            job_template_describe='tensorflow2.3分布式模型评估',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
            NO_RESOURCE_CHECK=true
            TASK_RESOURCE_CPU=4
            TASK_RESOURCE_MEMORY=4G
            TASK_RESOURCE_GPU=0
            '''.strip().replace(' ',''),
            job_template_expand={
                "index": 5,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/tf_distributed_evaluation"
            },

            job_template_args={
                "shell": {
                    "--job": {
                        "type": "json",
                        "item_type": "str",
                        "label": "模型评估，json配置",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "模型评估，json配置",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--upstream-output-file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "上游输出文件",
                        "require": 0,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "上游输出文件",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )


        # 注册tf 模型离线推理分布式
        create_template(
            repository_id=repository.id,
            project_name='tf分布式',
            image_name='ai.tencentmusic.com/tme-public/tf_model_offline_predict:latest',
            image_describe='tf模型离线推理',
            job_template_name='tf-model-offline-predict',
            job_template_describe='tf模型离线推理',
            job_template_command='',
            job_template_volume='',
            job_template_account='',
            job_template_env='',
            job_template_expand={
                "index": 6,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/tf_model_offline_predict"
            },

            job_template_args={
                "shell": {
                    "--job": {
                        "type": "json",
                        "item_type": "str",
                        "label": "json配置",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "json配置",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--upstream-output-file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "上游输出文件",
                        "require": 0,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "上游输出文件",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )




        # 注册pytorch分布式
        create_template(
            repository_id=repository.id,
            project_name='pytorch分布式',
            image_name='ai.tencentmusic.com/tme-public/pytorch_distributed_train_k8s:20201010',
            image_describe='pytorch分布式训练',
            job_template_name='pytorchjob-train',
            job_template_describe='pytorch 分布式训练',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
            NO_RESOURCE_CHECK=true
            TASK_RESOURCE_CPU=2
            TASK_RESOURCE_MEMORY=4G
            TASK_RESOURCE_GPU=0
            '''.strip().replace(' ',''),
            job_template_expand={
                "index": 2,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/pytorch_distributed_train_k8s"
            },

            job_template_args={
                "shell": {
                    "--image": {
                        "type": "str",
                        "item_type": "str",
                        "label": "worker镜像，直接运行你代码的环境镜像",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.6",
                        "placeholder": "",
                        "describe": "worker镜像，直接运行你代码的环境镜像 <a href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--working_dir": {
                        "type": "str",
                        "item_type": "str",
                        "label": "命令的启动目录",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "/mnt/xxx/pytorchjob/",
                        "placeholder": "",
                        "describe": "命令的启动目录",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--command": {
                        "type": "str",
                        "item_type": "str",
                        "label": "启动命令，例如 python3 xxx.py",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "启动命令，例如 python3 xxx.py",
                        "describe": "启动命令，例如 python3 xxx.py",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--num_worker": {
                        "type": "int",
                        "item_type": "str",
                        "label": "分布式训练worker的数目",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "3",
                        "placeholder": "分布式训练worker的数目",
                        "describe": "分布式训练worker的数目",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )




        # 注册分布式媒体文件下载
        create_template(
            repository_id=repository.id,
            project_name='多媒体类模板',
            image_name='ai.tencentmusic.com/tme-public/video-audio:20210601',
            image_describe='分布式媒体文件处理',
            job_template_name='media-download',
            job_template_describe='分布式下载媒体文件',
            job_template_command='python start_download.py',
            job_template_volume='2G(memory):/dev/shm',
            job_template_account='kubeflow-pipeline',
            job_template_expand={
                "index": 1,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/video-audio"
            },
            job_template_args={
                "shell": {
                    "--num_worker": {
                        "type": "str",
                        "item_type": "str",
                        "label": "分布式任务的worker数目",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "3",
                        "placeholder": "分布式任务的worker数目",
                        "describe": "分布式任务的worker数目",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--download_type": {
                        "type": "enum",
                        "item_type": "str",
                        "label": "下载类型",
                        "require": 1,
                        "choice": [
                            "url"
                        ],
                        "range": "",
                        "default": "url",
                        "placeholder": "",
                        "describe": "下载类型",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--input_file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "下载信息文件地址",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "下载信息文件地址<br>url类型，每行格式：$url $local_path",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }

        )

        # 注册分布式视频采帧
        create_template(
            repository_id=repository.id,
            project_name='多媒体类模板',
            image_name='ai.tencentmusic.com/tme-public/video-audio:20210601',
            image_describe='分布式媒体文件处理',
            job_template_name='video-img',
            job_template_describe='视频提取图片(分布式版)',
            job_template_command='python start_video_img.py',
            job_template_volume='2G(memory):/dev/shm',
            job_template_account='kubeflow-pipeline',
            job_template_expand={
                "index": 2,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/video-audio"
            },
            job_template_args={
                "shell": {
                    "--num_workers": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "worker数量",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--input_file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "配置文件地址，每行格式：<br>$local_video_path $des_img_dir $frame_rate",
                        "describe": "配置文件地址，每行格式：<br>$local_video_path $des_img_dir $frame_rate",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }

        )

        # 注册分布式音频提取模板
        create_template(
            repository_id=repository.id,
            project_name='多媒体类模板',
            image_name='ai.tencentmusic.com/tme-public/video-audio:20210601',
            image_describe='分布式媒体文件处理',
            job_template_name='video-audio',
            job_template_describe='视频提取音频(分布式版)',
            job_template_command='python start_video_audio.py',
            job_template_volume='2G(memory):/dev/shm',
            job_template_account='kubeflow-pipeline',
            job_template_expand={
                "index": 3,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/video-audio"
            },
            job_template_args={
                "shell": {
                    "--num_workers": {
                        "type": "str",
                        "item_type": "str",
                        "label": "worker数量",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "worker数量",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--input_file": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "",
                        "describe": "配置文件地址，每行格式：<br>$local_video_path $des_audio_path",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }

        )





        # 注册 kaldi 模型离线推理分布式
        create_template(
            repository_id=repository.id,
            project_name='多媒体类模板',
            image_name='ai.tencentmusic.com/tme-public/kaldi_distributed_on_volcano:v2',
            image_describe='kaldi音频分布式',
            job_template_name='kaldi-distributed-on-volcanojob',
            job_template_describe='kaldi音频分布式训练',
            job_template_command='',
            job_template_volume='4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
            NO_RESOURCE_CHECK=true
            TASK_RESOURCE_CPU=4
            TASK_RESOURCE_MEMORY=4G
            TASK_RESOURCE_GPU=0
            ''',
            job_template_expand={
                "index": 4,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/kaldi_distributed_on_volcanojob"
            },

            job_template_args={
                "shell": {
                    "--working_dir": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "启动目录",
                        "describe": "启动目录",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--user_cmd": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "./run.sh",
                        "placeholder": "启动命令",
                        "describe": "启动命令",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--num_worker": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "2",
                        "placeholder": "worker数量",
                        "describe": "worker数量",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--image": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "ai.tencentmusic.com/tme-public/kaldi_distributed_worker:v1",
                        "placeholder": "",
                        "describe": "worker镜像，直接运行你代码的环境镜像 <a href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )

        # 分布式离线推理
        create_template(
            repository_id=repository.id,
            project_name='多媒体类模板',
            image_name='ai.tencentmusic.com/tme-public/volcano:offline-predict-20220101',
            image_describe='分布式离线推理',
            job_template_name='model-offline-predict',
            job_template_describe='分布式离线推理',
            job_template_command='',
            job_template_volume='kubernetes-config(configmap):/root/.kube',
            job_template_account='kubeflow-pipeline',
            job_template_env='''
                NO_RESOURCE_CHECK=true
                TASK_RESOURCE_CPU=4
                TASK_RESOURCE_MEMORY=4G
                TASK_RESOURCE_GPU=0
                ''',
            job_template_expand={
                "index": 8,
                "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/model_offline_predict"
            },

            job_template_args={
                "shell": {
                    "--working_dir": {
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
                    "--command": {
                        "type": "str",
                        "item_type": "str",
                        "label": "环境安装和任务启动命令",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "/mnt/xx/../start.sh",
                        "placeholder": "",
                        "describe": "环境安装和任务启动命令",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--num_worker": {
                        "type": "str",
                        "item_type": "str",
                        "label": "占用机器个数",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "3",
                        "placeholder": "",
                        "describe": "占用机器个数",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    },
                    "--image": {
                        "type": "str",
                        "item_type": "str",
                        "label": "",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.6",
                        "placeholder": "",
                        "describe": "worker镜像，直接运行你代码的环境镜像<a href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            }
        )



    except Exception as e:
        print(e)



    # 添加demo 服务
    def create_service(project_name,service_name,service_describe,image_name,command,env,resource_mem='2G',resource_cpu='2',port='80'):
        service = db.session.query(Service).filter_by(name=service_name).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='org').first()
        if service is None and project:
            try:
                service = Service()
                service.name = service_name
                service.label=service_describe
                service.created_by_fk=1
                service.changed_by_fk=1
                service.project_id=project.id
                service.images=image_name
                service.command = command
                service.env='\n'.join([x.strip() for x in env.split('\n') if x.split()])
                service.port = port
                db.session.add(service)
                db.session.commit()
            except Exception as e:
                db.session.rollback()

    try:
        # 部署mysql-ui
        create_service(
            project_name='public',
            service_name='mysql-ui',
            service_describe='可视化编辑mysql数据库',
            image_name='ai.tencentmusic.com/tme-public/phpmyadmin',
            command='',
            env='''
            PMA_HOST=mysql-service.infra
            PMA_PORT=3306
            PMA_USER=root 
            PMA_PASSWORD=admin
            ''',
            port='80'
        )

        # 部署redis-ui
        create_service(
            project_name='public',
            service_name='redis-ui',
            service_describe='可视化编辑redis数据库',
            image_name='ai.tencentmusic.com/tme-public/patrikx3:latest',
            command='',
            env='''
            REDIS_NAME=default
            REDIS_HOST=redis-master.infra
            REDIS_PORT=6379
            REDIS_PASSWORD=admin
            ''',
            port='7843'
        )

        # 部署mongo ui
        create_service(
            project_name='public',
            service_name='mongo-express',
            service_describe='可视化编辑mongo数据库',
            image_name='mongo-express:0.54.0',
            command='',
            env='''
            ME_CONFIG_MONGODB_SERVER=xx.xx.xx.xx
            ME_CONFIG_MONGODB_PORT=xx
            ME_CONFIG_MONGODB_ENABLE_ADMIN=true
            ME_CONFIG_MONGODB_ADMINUSERNAME=xx
            ME_CONFIG_MONGODB_ADMINPASSWORD=xx
            ME_CONFIG_MONGODB_AUTH_DATABASE=xx
            ME_CONFIG_MONGODB_AUTH_USERNAME=xx
            ME_CONFIG_MONGODB_AUTH_PASSWORD=xx
            VCAP_APP_HOST=0.0.0.0
            VCAP_APP_PORT=8081
            ME_CONFIG_OPTIONS_EDITORTHEME=ambiance
            ''',
            port='8081'
        )


        # 部署neo4j
        create_service(
            project_name='public',
            service_name='neo4j',
            service_describe='可视化编辑图数据库neo4j',
            image_name='ai.tencentmusic.com/tme-public/neo4j:4.4',
            command='',
            env='''
            NEO4J_AUTH=neo4j/admin
            ''',
            port='7474,7687'
        )

        # 部署jaeger链路追踪
        create_service(
            project_name='public',
            service_name='jaeger',
            service_describe='jaeger链路追踪',
            image_name='jaegertracing/all-in-one:1.29',
            command='',
            env='',
            port='5775,16686'
        )


    except Exception as e:
        print(e)

@app.cli.command('init_db')
@pysnooper.snoop()
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



