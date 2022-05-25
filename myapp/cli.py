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
                        "describe": "worker镜像，直接运行你代码的环境镜像 <a href='https://docs.qq.com/doc/DU0ptZEpiSmtMY1JT'>基础镜像</a>",
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


        #
        # # 注册推理服务
        # create_template(
        #     repository_id=repository.id,
        #     project_name='模型服务化',
        #     image_name='ai.tencentmusic.com/tme-public/cube-service-deploy:latest',
        #     image_describe='模型部署推理服务',
        #     job_template_name='deploy-service',
        #     job_template_describe='模型部署推理服务',
        #     job_template_command='',
        #     job_template_volume='',
        #     job_template_account='',
        #     job_template_env='',
        #     job_template_expand={
        #         "index": 6,
        #         "help_url": "https://github.com/tencentmusic/cube-studio/tree/master/job-template/job/deploy-service"
        #     },
        #
        #     job_template_args={
        #         "shell": {
        #             "--service_type": {
        #                 "type": "str",
        #                 "item_type": "str",
        #                 "label": "模型服务类型",
        #                 "require": 1,
        #                 "choice": ['service','tfserving','torch-server','onnxruntime','triton-server'],
        #                 "range": "",
        #                 "default": "service",
        #                 "placeholder": "",
        #                 "describe": "模型服务类型",
        #                 "editable": 1,
        #                 "condition": "",
        #                 "sub_args": {}
        #             },
        #             "--project_name": {
        #                 "type": "str",
        #                 "item_type": "str",
        #                 "label": "项目组名称",
        #                 "require": 0,
        #                 "choice": [],
        #                 "range": "",
        #                 "default": "public",
        #                 "placeholder": "",
        #                 "describe": "项目组名称",
        #                 "editable": 1,
        #                 "condition": "",
        #                 "sub_args": {}
        #             },
        #             "--model_name": {
        #                 "type": "str",
        #                 "item_type": "str",
        #                 "label": "模型名",
        #                 "require": 0,
        #                 "choice": [],
        #                 "range": "",
        #                 "default": "",
        #                 "placeholder": "",
        #                 "describe": "模型名",
        #                 "editable": 1,
        #                 "condition": "",
        #                 "sub_args": {}
        #             },
        #             "--model_version": {
        #                 "type": "str",
        #                 "item_type": "str",
        #                 "label": "模型版本号",
        #                 "require": 0,
        #                 "choice": [],
        #                 "range": "",
        #                 "default": "",
        #                 "placeholder": "",
        #                 "describe": "模型版本号",
        #                 "editable": 1,
        #                 "condition": "",
        #                 "sub_args": {}
        #             },
        #             "--images": {
        #                 "type": "str",
        #                 "item_type": "str",
        #                 "label": "推理服务镜像",
        #                 "require": 0,
        #                 "choice": [],
        #                 "range": "",
        #                 "default": "",
        #                 "placeholder": "",
        #                 "describe": "推理服务镜像",
        #                 "editable": 1,
        #                 "condition": "",
        #                 "sub_args": {}
        #             },
        #             "--model_path": {
        #                 "type": "str",
        #                 "item_type": "str",
        #                 "label": "模型地址",
        #                 "require": 0,
        #                 "choice": [],
        #                 "range": "",
        #                 "default": "",
        #                 "placeholder": "",
        #                 "describe": "模型地址",
        #                 "editable": 1,
        #                 "condition": "",
        #                 "sub_args": {}
        #             }
        #         }
        #     }
        # )


    except Exception as e:
        print(e)



    # 创建demo pipeline
    def create_pipeline(tasks,pipeline):

        # 创建pipeline
        pipeline_model = db.session.query(Pipeline).filter_by(name=pipeline['name']).first()
        org_project = db.session.query(Project).filter_by(name=pipeline['project']).filter_by(type='org').first()
        if pipeline_model is None and org_project:
            try:
                pipeline_model = Pipeline()
                pipeline_model.name = pipeline['name']
                pipeline_model.describe = pipeline['describe']
                pipeline_model.dag_json=json.dumps(pipeline['dag_json'])
                pipeline_model.created_by_fk = 1
                pipeline_model.changed_by_fk = 1
                pipeline_model.project_id = org_project.id
                pipeline_model.expand = json.dumps(pipeline['expand'])
                db.session.add(pipeline_model)
                db.session.commit()
            except Exception as e:
                db.session.rollback()

        # 创建task
        for task in tasks:
            task_model = db.session.query(Task).filter_by(name=task['name']).filter_by(pipeline_id=pipeline_model.id).first()
            job_template = db.session.query(Job_Template).filter_by(name=task['job_templete']).first()
            if task_model is None and job_template:
                try:
                    task_model = Task()
                    task_model.name = task['name']
                    task_model.label = task['label']
                    task_model.working_dir = task.get('working_dir','')
                    task_model.command = task.get('command', '')
                    task_model.args = json.dumps(task['args'])
                    task_model.volume_mount = task['volume_mount']
                    task_model.resource_memory = task['resource_memory']
                    task_model.resource_cpu = task['resource_cpu']
                    task_model.resource_gpu = task['resource_gpu']
                    task_model.created_by_fk = 1
                    task_model.changed_by_fk = 1
                    task_model.pipeline_id = pipeline_model.id
                    task_model.job_template_id = job_template.id
                    db.session.add(task_model)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()


    try:
        pipeline={
            "name":"imageAI",
            "describe":"图像预测+物体检测+视频跟踪",
            "dag_json":{},
            "project":"public",
            "expand":{
                "demo":"true",
                "img":"https://user-images.githubusercontent.com/20157705/170216784-91ac86f7-d272-4940-a285-0c27d6f6cd96.jpg"
            }
        }
        tasks=[]
        create_pipeline(pipeline=pipeline,tasks=tasks)
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



