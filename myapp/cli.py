#!/usr/bin/env python
import shutil
from datetime import datetime
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import json
from myapp import app, appbuilder, db, security_manager
from myapp.models.model_team import Project, Project_User
from myapp.models.model_job import Repository, Images, Job_Template, Pipeline, Task
from myapp.models.model_dataset import Dataset
from myapp.models.model_serving import Service, InferenceService
from myapp.models.model_train_model import Training_Model
from myapp.models.model_notebook import Notebook
import uuid
import os
import importlib
import traceback
conf = app.config
import pysnooper

def create_app(script_info=None):
    return app


@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db)

# @pysnooper.snoop()
def replace_git(dir_path):
    files = os.listdir(dir_path)
    for file_name in files:
        file_path = os.path.join(dir_path,file_name)
        if os.path.isfile(file_path) and '.json' in file_name:
            content = open(file_path).read()
            content = content.replace('https://github.com/tencentmusic/cube-studio/tree/master',conf.get('GIT_URL',''))
            file = open(file_path,mode='w')
            file.write(content)
            file.close()



# https://dormousehole.readthedocs.io/en/latest/cli.html
@app.cli.command('init')
# @pysnooper.snoop()
def init():
    try:
        """Inits the Myapp application"""
        appbuilder.add_permissions(update_perms=True)  # update_perms为true才会检测新权限
        security_manager.sync_role_definitions()
    except Exception as e:
        print(e)

    init_dir='myapp/init' if conf.get('BABEL_DEFAULT_LOCALE','zh')=='zh' else "myapp/init-en"
    replace_git(init_dir)
    # 初始化创建项目组
    try:

        def add_project(project_type, name, describe, expand={}):
            if not expand:
                expand={
                    "org": "public"
                }
            print('add project',project_type,name,describe)
            project = db.session.query(Project).filter_by(name=name).filter_by(type=project_type).first()
            if project is None:
                try:
                    project = Project()
                    project.type = project_type
                    project.name = name
                    project.describe = describe
                    project.expand = json.dumps(expand, ensure_ascii=False, indent=4)
                    db.session.add(project)
                    db.session.commit()

                    project_user = Project_User()
                    project_user.project = project
                    project_user.role = 'creator'
                    project_user.user_id = 1
                    db.session.add(project_user)
                    db.session.commit()
                    print('add project %s' % name)
                except Exception as e:
                    print(e)
                    db.session.rollback()

        # 添加一些默认的记录
        add_project('org', 'public', __('公共项目组'),expand={'cluster':'dev','org':'public'})
        add_project('org', __('推荐中心'), __('推荐项目组'),expand={'cluster':'dev','org':'public'})
        add_project('org', __('搜索中心'), __('搜索项目组'),expand={'cluster':'dev','org':'public'})
        add_project('org', __('广告中心'), __('广告项目组'),expand={'cluster':'dev','org':'public'})
        add_project('org', __('安全中心'), __('安全项目组'),expand={'cluster':'dev','org':'public'})
        add_project('org', __('多媒体中心'), __('多媒体项目组'),expand={'cluster':'dev','org':'public'})

        add_project('job-template', __('基础命令'), __('python/bash等直接在服务器命令行中执行命令的模板'), {"index": 1})
        add_project('job-template', __('数据导入导出'), __('集群与用户机器或其他集群之间的数据迁移'), {"index": 2})
        add_project('job-template', __('数据预处理'), __('结构化话数据特征处理'), {"index": 3})
        add_project('job-template', __('数据处理工具'), __('数据的单机或分布式处理任务,ray/spark/hadoop/volcanojob'), {"index": 4})
        add_project('job-template', __('特征处理'), __('特征处理相关功能'), {"index": 5})
        add_project('job-template', __('机器学习框架'), __('传统机器学习框架，sklearn'), {"index": 6})
        add_project('job-template', __('机器学习算法'), __('传统机器学习，lr/决策树/gbdt/xgb/fm等'), {"index": 7})
        add_project('job-template', __('深度学习'), __('深度框架训练，tf/pytorch/mxnet/mpi/horovod/kaldi等'), {"index": 8})
        add_project('job-template', __('分布式加速'), __('分布式训练加速框架'), {"index": 9})
        add_project('job-template', __('tf分布式'), __('tf相关的训练，模型校验，离线预测等功能'), {"index": 10})
        add_project('job-template', __('pytorch分布式'), __('pytorch相关的训练，模型校验，离线预测等功能'), {"index": 11})
        add_project('job-template', __('模型处理'), __('模型压缩转换处理相关的组件模板'), {"index": 13})
        add_project('job-template', __('模型服务化'), __('模型服务化部署相关的组件模板'), {"index": 14})
        add_project('job-template', __('推荐类模板'), __('推荐领域常用的任务模板'), {"index": 15})
        add_project('job-template', __('搜索类模板'), __('搜索领域常用的任务模板'), {"index": 16})
        add_project('job-template', __('广告类模板'), __('广告领域常用的任务模板'), {"index": 17})
        add_project('job-template', __('多媒体类模板'), __('音视频图片文本常用的任务模板'), {"index": 18})
        add_project('job-template', __('机器视觉'), __('视觉类相关模板'), {"index": 19})
        add_project('job-template', __('听觉'), __('听觉类相关模板'), {"index": 20})
        add_project('job-template', __('自然语言'), __('自然语言类相关模板'), {"index": 21})
        add_project('job-template', __('大模型'), __('大模型相关模板'), {"index": 22})

    except Exception as e:
        print(e)


    # @pysnooper.snoop()
    def create_template(repository_id, project_name, image_name, image_describe, job_template_name,
                        job_template_old_names=[], job_template_describe='',job_template_workdir='', job_template_command='',
                        job_template_args=None, job_template_volume='', job_template_account='',
                        job_template_expand=None, job_template_env='', gitpath='',**kwargs):
        if not repository_id:
            return
        images = db.session.query(Images).filter_by(name=image_name).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='job-template').first()
        if images is None and project:
            try:
                images = Images()
                images.name = image_name
                images.describe = image_describe
                images.created_by_fk = 1
                images.changed_by_fk = 1
                images.project_id = project.id
                images.repository_id = repository_id
                images.gitpath = gitpath
                db.session.add(images)
                db.session.commit()
                print('add images %s' % image_name)
            except Exception as e:
                print(e)
                db.session.rollback()

        job_template = db.session.query(Job_Template).filter_by(name=job_template_name).first()
        if not job_template:
            for old_name in job_template_old_names:
                job_template = db.session.query(Job_Template).filter_by(name=old_name).first()
                if job_template:
                    break

        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='job-template').first()
        if project and images.id:
            if job_template is None:
                try:
                    job_template = Job_Template()
                    job_template.name = job_template_name.replace('_', '-')
                    job_template.describe = job_template_describe
                    job_template.version = kwargs.get('job_template_version','Release')
                    job_template.workdir = job_template_workdir
                    job_template.entrypoint = job_template_command
                    job_template.volume_mount = job_template_volume
                    job_template.accounts = job_template_account
                    job_template_expand['source'] = "github"
                    job_template.expand = json.dumps(job_template_expand, indent=4, ensure_ascii=False) if job_template_expand else '{}'
                    job_template.created_by_fk = 1
                    job_template.changed_by_fk = 1
                    job_template.project_id = project.id
                    job_template.images_id = images.id
                    job_template.env = job_template_env
                    job_template.args = json.dumps(job_template_args, indent=4, ensure_ascii=False) if job_template_args else '{}'
                    db.session.add(job_template)
                    db.session.commit()
                    print('add job_template %s' % job_template_name.replace('_', '-'))
                except Exception as e:
                    print(e)
                    db.session.rollback()
            else:
                pass
                # try:
                #     job_template.name = job_template_name.replace('_', '-')
                #     job_template.describe = job_template_describe
                #     job_template.entrypoint = job_template_command
                #     job_template.volume_mount = job_template_volume
                #     job_template.accounts = job_template_account
                #     job_template_expand['source'] = "github"
                #     job_template.expand = json.dumps(job_template_expand, indent=4, ensure_ascii=False) if job_template_expand else '{}'
                #     job_template.created_by_fk = 1
                #     job_template.changed_by_fk = 1
                #     job_template.project_id = project.id
                #     job_template.images_id = images.id
                #     job_template.version = 'Release'
                #     job_template.env = job_template_env
                #     job_template.args = json.dumps(job_template_args, indent=4, ensure_ascii=False) if job_template_args else '{}'
                #     db.session.commit()
                #     print('update job_template %s' % job_template_name.replace('_', '-'))
                # except Exception as e:
                #     print(e)
                #     db.session.rollback()

    # 初始化创建仓库镜像模板任务流
    try:
        print('begin init repository')
        repository = db.session.query(Repository).filter_by(name='hubsecret').first()
        if repository is None:
            try:
                repository = Repository()
                repository.name = 'hubsecret'
                repository.server='registry.docker-cn.com'
                repository.user = 'yourname'
                repository.password = 'yourpassword'
                repository.hubsecret = 'hubsecret'
                repository.created_by_fk = 1
                repository.changed_by_fk = 1
                db.session.add(repository)
                db.session.commit()
                print('add repository hubsecret')
            except Exception as e:
                print(e)
                db.session.rollback()


        print('begin init job_templates')
        init_file = os.path.join(init_dir,'init-job-template.json')
        if os.path.exists(init_file):
            job_templates = json.load(open(init_file, mode='r'))
            for job_template_name in job_templates:
                try:
                    job_template = job_templates[job_template_name]
                    job_template['repository_id'] = repository.id
                    create_template(**job_template)
                except Exception as e1:
                    print(e1)

    except Exception as e:
        print(e)

    # 创建demo pipeline
    import pysnooper
    # @pysnooper.snoop()
    def create_pipeline(tasks, pipeline):
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
                pipeline_model.dag_json = json.dumps(pipeline['dag_json'], indent=4, ensure_ascii=False).replace('_', '-')
                pipeline_model.created_by_fk = 1
                pipeline_model.changed_by_fk = 1
                pipeline_model.project_id = org_project.id
                pipeline_model.global_env = pipeline.get('global_env','')
                pipeline_model.parameter = json.dumps(pipeline.get('parameter', {}), indent=4, ensure_ascii=False)
                pipeline_model.expand = json.dumps(pipeline.get('expand', {}), indent=4, ensure_ascii=False)
                db.session.add(pipeline_model)
                db.session.commit()
                print('add pipeline %s' % pipeline['name'])
            except Exception as e:
                print(e)
                db.session.rollback()
        else:
            return
            # pipeline_model.describe = pipeline['describe']
            # pipeline_model.dag_json = json.dumps(pipeline['dag_json'], indent=4, ensure_ascii=False).replace('_', '-')
            # pipeline_model.created_by_fk = 1
            # pipeline_model.changed_by_fk = 1
            # pipeline_model.global_env = pipeline['global_env']
            # pipeline_model.project_id = org_project.id
            # pipeline_model.parameter = json.dumps(pipeline.get('parameter', {}))
            # pipeline_model.expand = json.dumps(pipeline.get('expand', {}), indent=4, ensure_ascii=False)
            # print('update pipeline %s' % pipeline['name'])
            # db.session.commit()

        # 创建task
        for task in tasks:
            task_model = db.session.query(Task).filter_by(name=task['name']).filter_by(pipeline_id=pipeline_model.id).first()
            job_template = db.session.query(Job_Template).filter_by(name=task['job_templete']).first()
            if task_model is None and job_template:
                try:
                    task_model = Task()
                    task_model.name = task['name'].replace('_', '-')
                    task_model.label = task['label']
                    task_model.args = json.dumps(task['args'], indent=4, ensure_ascii=False)
                    task_model.volume_mount = task.get('volume_mount', '')
                    task_model.resource_memory = task.get('resource_memory', '2G')
                    task_model.resource_cpu = task.get('resource_cpu', '2')
                    task_model.resource_gpu = task.get('resource_gpu', '0')
                    task_model.resource_rdma = task.get('resource_rdma', '0')
                    task_model.created_by_fk = 1
                    task_model.changed_by_fk = 1
                    task_model.pipeline_id = pipeline_model.id
                    task_model.job_template_id = job_template.id
                    db.session.add(task_model)
                    db.session.commit()
                    print('add task %s' % task['name'])
                except Exception as e:
                    print(e)
                    # # traceback.print_exc()
                    db.session.rollback()
            else:
                pass
                # task_model.label = task['label']
                # task_model.args = json.dumps(task['args'], indent=4, ensure_ascii=False)
                # task_model.volume_mount = task.get('volume_mount', '')
                # task_model.node_selector = task.get('node_selector', 'cpu=true,train=true,org=public')
                # task_model.retry = int(task.get('retry', 0))
                # task_model.timeout = int(task.get('timeout', 0))
                # task_model.resource_memory = task.get('resource_memory', '2G')
                # task_model.resource_cpu = task.get('resource_cpu', '2')
                # task_model.resource_gpu = task.get('resource_gpu', '0')
                # task_model.created_by_fk = 1
                # task_model.changed_by_fk = 1
                # task_model.pipeline_id = pipeline_model.id
                # task_model.job_template_id = job_template.id
                # print('update task %s' % task['name'])
                # db.session.commit()

        pipeline_model.dag_json = pipeline_model.fix_dag_json()  # 修正 dag_json
        # 没有设置位置的时候，修正pipeline
        if not pipeline_model.expand or not json.loads(pipeline_model.expand):
            pipeline_model.expand = json.dumps(pipeline_model.fix_expand(), indent=4, ensure_ascii=False)  # 修正 前端expand字段缺失
            pipeline_model.expand = json.dumps(pipeline_model.fix_position(), indent=4, ensure_ascii=False)  # 修正 节点中心位置到视图中间
            db.session.commit()
            # 自动排版
            db_tasks = pipeline_model.get_tasks(db.session)
            if db_tasks:
                try:
                    tasks = {}
                    for task in db_tasks:
                        tasks[task.name] = task.to_json()

                    from myapp.utils import core
                    expand = core.fix_task_position(pipeline_model.to_json(), tasks, json.loads(pipeline_model.expand))
                    pipeline_model.expand = json.dumps(expand, indent=4, ensure_ascii=False)
                    db.session.commit()
                except Exception as e:
                    print(e)
                    # traceback.print_exc()
        else:
            # 把expand中的任务名换成任务id
            pipeline_expands = json.loads(pipeline_model.expand)
            tasks = pipeline_model.get_tasks()
            tasks_ids = {}
            for task in tasks:
                tasks_ids[str(task.name)] = task

            expands = []
            for exp in pipeline_expands:
                # 节点信息
                if 'source' in exp:
                    exp = {
                        "source": str(tasks_ids[exp['source']].id),
                        "arrowHeadType": "arrow",
                        "target": str(tasks_ids[exp['target']].id),
                        "id": "logic__edge-%snull-%snull" % (tasks_ids[exp['source']].id, tasks_ids[exp['target']].id)
                    }
                # 连接线信息
                else:
                    exp = {
                        "id": str(tasks_ids[exp['id']].id),
                        "type": "dataSet",
                        "position": {
                            "x": int(exp['position']['x']),
                            "y": int(exp['position']['y'])
                        },
                        "data": {
                            "info": {
                                "describe": tasks_ids[exp['id']].job_template.describe
                            },
                            "name": tasks_ids[exp['id']].name,
                            "label": tasks_ids[exp['id']].label
                        }
                    }
                expands.append(exp)
            pipeline_model.expand = json.dumps(expands)
            db.session.commit()
            pass
    try:
        print('begin init pipeline')
        init_file = os.path.join(init_dir,'init-pipeline.json')
        if os.path.exists(init_file):
            pipelines = json.load(open(init_file, mode='r'))
            for pipeline_name in pipelines:
                try:
                    pipeline = pipelines[pipeline_name]['pipeline']
                    tasks = pipelines[pipeline_name]['tasks']
                    create_pipeline(pipeline=pipeline, tasks=tasks)
                    print('add pipeline %s' % pipeline_name)
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)
        # traceback.print_exc()

    # 从目录中添加示例 pipeline
    try:
        print('begin init pipeline example')

        pipelines = os.listdir('myapp/example/pipeline/')
        for pipeline_name in pipelines:
            if os.path.isdir(os.path.join('myapp/example/pipeline/',pipeline_name)):
                try:
                    pipeline_path = os.path.join('myapp/example/pipeline/',pipeline_name,'pipeline.json')
                    init_path = os.path.join('myapp/example/pipeline/', pipeline_name, 'init.py')
                    if os.path.exists(pipeline_path):
                        pipeline = json.load(open(pipeline_path))
                        tasks = pipeline['tasks']
                        pipeline = pipeline['pipeline']

                        create_pipeline(pipeline=pipeline, tasks=tasks)

                    # 环境要求比较复杂，可以直接在notebook里面初始化
                    os.makedirs('/data/k8s/kubeflow/pipeline/workspace/admin/pipeline/example/',exist_ok=True)
                    # shutil.copy2(f'myapp/example/pipeline/{pipeline_name}','/data/k8s/kubeflow/pipeline/workspace/admin/pipeline/example/')
                    if not os.path.exists(f'/data/k8s/kubeflow/pipeline/workspace/admin/pipeline/example/{pipeline_name}'):
                        shutil.copytree(f'myapp/example/pipeline/{pipeline_name}', f'/data/k8s/kubeflow/pipeline/workspace/admin/pipeline/example/{pipeline_name}')
                    # if os.path.exists(init_path):
                    #     try:
                    #         params = importlib.import_module(f'myapp.example.pipeline.{pipeline_name}.init')
                    #         init_func = getattr(params, 'init')
                    #         init_func()
                    #     except Exception as e:
                    #         print(e)
                    #         # traceback.print_exc()

                    print('add job template using example %s' % pipeline_name)
                except Exception as e1:
                    print(e1)
                # traceback.print_exc()
    except Exception as e:
        print(e)
        # traceback.print_exc()


    # 添加 demo 推理 服务
    def create_dataset(**kwargs):
        dataset = db.session.query(Dataset).filter_by(name=kwargs['name']).first()
        if not dataset:
            try:
                dataset = Dataset()
                dataset.name = kwargs['name']
                dataset.field = kwargs.get('field', '')
                dataset.version = 'latest'
                dataset.label = kwargs.get('label', '')
                dataset.status = kwargs.get('status', '')
                dataset.describe = kwargs.get('describe', '')
                dataset.url = kwargs.get('url', '')
                dataset.source = kwargs.get('source', '')
                dataset.industry = kwargs.get('industry', '')
                dataset.source_type = kwargs.get('source_type', '')
                dataset.file_type = kwargs.get('file_type', '')
                dataset.research = kwargs.get('research', '')
                dataset.usage = kwargs.get('usage', '')
                dataset.years = kwargs.get('years', '')
                dataset.path = kwargs.get('path', '')
                dataset.duration = kwargs.get('duration', '')
                dataset.entries_num = kwargs.get('entries_num', '')
                dataset.price = kwargs.get('price', '')
                dataset.icon = kwargs.get('icon', '')
                dataset.storage_class = kwargs.get('storage_class', '')
                dataset.storage_size = kwargs.get('storage_size', '')
                dataset.download_url = kwargs.get('download_url', '')
                dataset.owner = 'admin'
                dataset.created_by_fk = 1
                dataset.changed_by_fk = 1
                db.session.add(dataset)
                db.session.commit()
                print('add dataset %s' % kwargs.get('name', ''))
            except Exception as e:
                print(e)
                # traceback.print_exc()
                db.session.rollback()

    try:
        print('begin init dataset')
        datasets = db.session.query(Dataset).all()  # 空白数据集才初始化
        if not datasets:
            import csv
            init_file = os.path.join(init_dir, 'init-dataset.csv')
            if os.path.exists(init_file):
                csv_reader = csv.reader(open(init_file, mode='r', encoding='utf-8-sig'))
                header = None
                for line in csv_reader:
                    if not header:
                        header = line
                        continue
                    data = dict(zip(header, line))
                    create_dataset(**data)

    except Exception as e:
        print(e)
        # traceback.print_exc()

    # 添加 示例 模型
    # @pysnooper.snoop()
    def create_train_model(name, describe, path, project_name, version, framework, api_type):
        train_model = db.session.query(Training_Model).filter_by(name=name).filter_by(version=version).filter_by(framework=framework).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='org').first()
        if not train_model and project:
            try:
                train_model = Training_Model()
                train_model.name = name
                train_model.describe = describe
                train_model.path = path
                train_model.project_id = project.id
                train_model.describe = describe
                train_model.version = version
                train_model.framework = framework
                train_model.api_type = api_type
                train_model.created_by_fk = 1
                train_model.changed_by_fk = 1
                train_model.run_id = 'random_run_id_' + uuid.uuid4().hex[:32]
                db.session.add(train_model)
                db.session.commit()
                print('add train model %s' % name)
            except Exception as e:
                print(e)
                # traceback.print_exc()
                db.session.rollback()

    try:
        print('begin init train_models')
        init_file = os.path.join(init_dir, 'init-train-model.json')
        if os.path.exists(init_file):
            train_models = json.load(open(init_file, mode='r'))
            for train_model_name in train_models:
                try:
                    train_model = train_models[train_model_name]
                    create_train_model(**train_model)
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)
        # traceback.print_exc()

    # 添加demo 服务
    # @pysnooper.snoop()
    def create_service(project_name, service_name, service_describe, image_name, command, env, resource_memory='2G',
                       resource_cpu='2', resource_gpu='0', ports='80', volume_mount='kubeflow-user-workspace(pvc):/mnt',
                       expand={}):
        service = db.session.query(Service).filter_by(name=service_name).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='org').first()
        if service is None and project:
            try:
                service = Service()
                service.name = service_name.replace('_', '-')
                service.label = service_describe
                service.created_by_fk = 1
                service.changed_by_fk = 1
                service.project_id = project.id
                service.images = image_name
                service.command = command
                service.resource_memory = resource_memory
                service.resource_cpu = resource_cpu
                service.resource_gpu = resource_gpu
                service.env = '\n'.join([x.strip() for x in env.split('\n') if x.split()])
                service.ports = ports
                service.volume_mount = volume_mount
                service.expand = json.dumps(expand, indent=4, ensure_ascii=False)
                db.session.add(service)
                db.session.commit()
                print('add service %s' % service_name)
            except Exception as e:
                print(e)
                # traceback.print_exc()
                db.session.rollback()

    try:
        print('begin init services')
        init_file = os.path.join(init_dir, 'init-service.json')
        if os.path.exists(init_file):
            services = json.load(open(init_file, mode='r'))
            for service_name in services:
                try:
                    service = services[service_name]
                    create_service(**service)
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)
        # traceback.print_exc()

    # 添加 demo 推理 服务
    # @pysnooper.snoop()
    def create_inference(project_name, service_name, service_describe, image_name, command, env, model_name, workdir='',
                         model_version='', model_path='', service_type='serving', resource_memory='2G',
                         resource_cpu='2', resource_gpu='0', host='', ports='80',
                         volume_mount='kubeflow-user-workspace(pvc):/mnt', metrics='', health='', inference_config='',
                         expand={}):
        service = db.session.query(InferenceService).filter_by(name=service_name).first()
        project = db.session.query(Project).filter_by(name=project_name).filter_by(type='org').first()
        if service is None and project:
            try:
                service = InferenceService()
                service.name = service_name.replace('_', '-')
                service.label = service_describe
                service.service_type = service_type
                service.model_name = model_name
                service.model_version = model_version if model_version else datetime.now().strftime('v%Y.%m.%d.1')
                service.model_path = model_path
                service.created_by_fk = 1
                service.changed_by_fk = 1
                service.project_id = project.id
                service.images = image_name
                service.resource_memory = resource_memory
                service.resource_cpu = resource_cpu
                service.resource_gpu = resource_gpu
                service.host = host
                service.working_dir = workdir
                service.command = command
                service.inference_config = inference_config
                service.env = '\n'.join([x.strip() for x in env.split('\n') if x.split()])
                service.ports = ports
                service.volume_mount = volume_mount
                service.metrics = metrics
                service.health = health
                service.expand = json.dumps(expand, indent=4, ensure_ascii=False)

                from myapp.views.view_inferenceserving import InferenceService_ModelView_base
                inference_class = InferenceService_ModelView_base()
                inference_class.src_item_json = {}
                inference_class.pre_add(service)

                db.session.add(service)
                db.session.commit()
                print('add inference %s' % service_name)
            except Exception as e:
                print(e)
                # traceback.print_exc()
                db.session.rollback()

    try:
        print('begin init inferences')
        init_file = os.path.join(init_dir, 'init-inference.json')
        if os.path.exists(init_file):
            inferences = json.load(open(init_file, mode='r'))
            for inference_name in inferences:
                try:
                    inference = inferences[inference_name]
                    create_inference(**inference)
                except Exception as e1:
                    print(e1)
    except Exception as e:
        print(e)
        # traceback.print_exc()

    def add_aihub(info_path):
        from myapp.models.model_aihub import Aihub
        if not os.path.exists(info_path):
            return
        aihubs = json.load(open(info_path, mode='r'))

        try:
            if len(aihubs) > 0:
                # dbsession.query(Aihub).delete()
                # dbsession.commit()
                print('add aihub ', end=' ')
                for data in aihubs:
                    name = data.get('name', '')
                    print(name, end=' ')
                    label = data.get('label', '')
                    describe = data.get('describe', '')
                    uuid = data.get('uuid', '')
                    if name and label and describe and uuid:
                        aihub = db.session.query(Aihub).filter_by(uuid=uuid).first()
                        if not aihub:
                            aihub = Aihub()
                        aihub.doc = data.get('doc', '')
                        aihub.name = name
                        aihub.label = label
                        aihub.describe = describe
                        aihub.field = data.get('field', '')
                        aihub.scenes = data.get('scenes', '')
                        aihub.type = data.get('type', '')
                        aihub.pic = data.get('pic', '')
                        aihub.status = data.get('status', '')
                        aihub.uuid = uuid
                        aihub.images = data.get('images', '')
                        aihub.version = data.get('version', '')
                        aihub.dataset = json.dumps(data.get('dataset', {}), indent=4, ensure_ascii=False)
                        aihub.notebook = json.dumps(data.get('notebook', {}), indent=4, ensure_ascii=False)
                        aihub.job_template = json.dumps(data.get('train', {}), indent=4, ensure_ascii=False)
                        aihub.pre_train_model = json.dumps(data.get('pre_train_model', {}), indent=4, ensure_ascii=False)
                        aihub.inference = json.dumps(data.get('inference', {}), indent=4, ensure_ascii=False)
                        aihub.service = json.dumps(data.get('service', {}), indent=4, ensure_ascii=False)
                        aihub.hot = int(data.get('hot', '0'))
                        aihub.price = int(data.get('price', '0'))
                        aihub.source = data.get('source', '')
                        if not aihub.id:
                            db.session.add(aihub)
                        db.session.commit()
        except Exception as e:
            print(e)
            # traceback.print_exc()

    # 添加aihub
    try:
        print('begin add aihub')
        init_file = os.path.join(init_dir, 'init-aihub.json')
        if os.path.exists(init_file):
            add_aihub(init_file)
    except Exception as e:
        print(e)
        # traceback.print_exc()

    # 复制cube-studio代码（aihub和sdk）
    from myapp.tasks.schedules import cp_cubestudio
    cp_cubestudio()

    def add_chat(chat_path):
        from myapp.models.model_chat import Chat
        if not os.path.exists(chat_path):
            return
        chats = json.load(open(chat_path, mode='r'))

        try:
            if len(chats) > 0:
                for data in chats:
                    # print(data)
                    name = data.get('name', '')
                    label = data.get('label', '')
                    if name and label:
                        chat = db.session.query(Chat).filter_by(name=name).first()
                        if not chat:
                            knowledge = data.get('knowledge', '')
                            if type(knowledge)==dict:
                                knowledge = json.dumps(knowledge,indent=4,ensure_ascii=False)
                            chat = Chat()
                            chat.doc = data.get('doc', '')
                            chat.name = name
                            chat.label = label
                            chat.icon = data.get('icon', '')
                            chat.session_num = int(data.get('session_num', '0'))
                            chat.chat_type = data.get('chat_type', 'text')
                            chat.hello = data.get('hello', '这里是cube-studio开源社区，请问有什么可以帮你的么？')
                            chat.tips = data.get('tips', '')
                            chat.prompt = data.get('prompt', '')
                            chat.knowledge = knowledge
                            chat.service_type = data.get('service_type', 'chatgpt3.5')
                            chat.service_config = json.dumps(data.get('service_config', {}), indent=4, ensure_ascii=False)
                            chat.owner = data.get('owner', 'admin')
                            chat.expand = json.dumps(data.get('expand', {}), indent=4,ensure_ascii=False)

                            if not chat.id:
                                db.session.add(chat)
                            db.session.commit()
        except Exception as e:
            print(e)
            # traceback.print_exc()

    try:
        print('begin add chat')
        init_file = os.path.join(init_dir, 'init-chat.json')
        if os.path.exists(init_file):
            add_chat(init_file)
    except Exception as e:
        print(e)
            # traceback.print_exc()
    # 添加chat
    # if conf.get('BABEL_DEFAULT_LOCALE','zh')=='zh':
    try:
        SQLALCHEMY_DATABASE_URI = os.getenv('MYSQL_SERVICE', '')
        if SQLALCHEMY_DATABASE_URI:
            import sqlalchemy.engine.url as url
            uri = url.make_url(SQLALCHEMY_DATABASE_URI)
            database = uri.database
            from myapp.models.model_metadata import Metadata_table
            tables = db.session.query(Metadata_table).all()
            if len(tables)==0:
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='project', owner='admin',describe='项目分组，模板分组，模型分组'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='project_user', owner='admin',describe='项目组用户'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='idex_query', owner='admin',describe='sqllab的查询记录'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='metadata_table', owner='admin',describe='离线库表管理'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='metadata_metric', owner='admin',describe='指标管理'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='dimension', owner='admin',describe='维表管理'))
                db.session.add(Metadata_table(app='cube-studio',db='kubeflow',table='dataset',owner='admin',describe='数据集市场'))

                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='repository', owner='admin',describe='docker仓库管理'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='docker', owner='admin', describe='在线docker镜像构建'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='images', owner='admin',describe='镜像管理'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='notebook', owner='admin', describe='notebook在线开发'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='etl_pipeline', owner='admin',describe='数据ETL的任务流管理'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='etl_task', owner='admin',describe='数据ETL的任务管理'))

                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='job_template', owner='admin',describe='任务模板'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='pipeline', owner='admin',describe='ml任务流'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='task', owner='admin', describe='ml任务管理'))

                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='run', owner='admin',describe='定时调度记录'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='workflow', owner='admin',describe='任务流实例'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='nni', owner='admin', describe='nni超参搜索'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='service', owner='admin',describe='内部服务管理'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='model', owner='admin', describe='模型管理'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='inferenceservice', owner='admin', describe='推理服务'))

                db.session.add(Metadata_table(app='cube-studio',db='kubeflow',table='aihub',owner='admin',describe='模型应用市场，打通自动化标注，一键开发，一键微调，一建部署'))
                db.session.add(Metadata_table(app='cube-studio',db='kubeflow',table='chat',owner='admin',describe='私有知识库，配置领域知识文档或qa文档，智能机器人问答'))
                db.session.add(Metadata_table(app='cube-studio',db='kubeflow',table='chat_log',owner='admin',describe='所有的聊天日志记录'))
                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='favorite', owner='admin',describe='收藏的数据记录'))

                db.session.add(Metadata_table(app='cube-studio', db='kubeflow', table='logs', owner='admin',describe='用户行为记录'))
                db.session.commit()
                print('添加离线表成功')

    except Exception as e:
        print(e)
        # traceback.print_exc()
    # 添加ETL pipeline
    try:
        from myapp.models.model_etl_pipeline import ETL_Pipeline
        tables = db.session.query(ETL_Pipeline).all()
        if len(tables) == 0:
            init_file = os.path.join(init_dir, 'init-etl-pipeline.json')
            if os.path.exists(init_file):
                pipelines = json.load(open(init_file, mode='r'))
                for pipeline in pipelines:
                    db.session.add(ETL_Pipeline(
                        project_id=1, created_by_fk=1,changed_by_fk=1,
                        name=pipeline.get('name',''), config=json.dumps(pipeline.get('config',{}),indent=4,ensure_ascii=False),
                        describe=pipeline.get('describe','pipeline example'),workflow=pipeline.get('workflow','airflow'),
                        dag_json=json.dumps(pipeline.get('dag_json',{}),indent=4,ensure_ascii=False)))
                    db.session.commit()
                    print('添加etl pipeline成功')
    except Exception as e:
        print(e)
        # traceback.print_exc()


    # 添加nni超参搜索
    try:
        from myapp.models.model_nni import NNI
        nni = db.session.query(NNI).all()
        if len(nni) == 0:
            init_file = os.path.join(init_dir, 'init-automl.json')
            if os.path.exists(init_file):
                nnis = json.load(open(init_file, mode='r'))
                for nni in nnis:
                    db.session.add(NNI(
                        project_id=1, created_by_fk=1,changed_by_fk=1,
                        job_type=nni.get('job_type','Job'),name=nni.get('name','test'+uuid.uuid4().hex[:4]),namespace=nni.get('namespace','automl'),
                        describe=nni.get('describe', ''),parallel_trial_count=nni.get('parallel_trial_count', 3),max_trial_count=nni.get('max_trial_count', 12),
                        objective_type=nni.get('objective_type', 'maximize'),objective_goal=nni.get('objective_goal', 0.99),objective_metric_name=nni.get('objective_metric_name', 'accuracy'),
                        algorithm_name=nni.get('algorithm_name','Random'), parameters=json.dumps(nni.get('parameters',{}),indent=4,ensure_ascii=False),
                        job_json=json.dumps(nni.get('job_json',{}),indent=4,ensure_ascii=False),
                        job_worker_image = nni.get('job_worker_image', conf.get('NNI_IMAGES','')),
                        working_dir=nni.get('working_dir', '/mnt/admin/nni/demo/'),
                        job_worker_command=nni.get('job_worker_command', 'python xx.py'),
                        resource_memory=nni.get('resource_memory', '1G'),
                        resource_cpu=nni.get('resource_cpu', '1'),
                        resource_gpu=nni.get('resource_gpu', '0'),
                    ))
                    db.session.commit()
                    print('添加etl pipeline成功')
    except Exception as e:
        print(e)
        # traceback.print_exc()


    # 添加镜像在线构建
    try:
        from myapp.models.model_docker import Docker
        docker = db.session.query(Docker).all()
        if len(docker) == 0:

            db.session.add(Docker(
                project_id=1,
                created_by_fk=1,
                changed_by_fk=1,
                describe='build python environment',
                base_image=conf.get('USER_IMAGE',''),
                target_image=conf.get("REPOSITORY_ORG",'')+'python:2023.06.19.1',
                need_gpu=False,
                consecutive_build=True,
                expand=json.dumps(
                {
                    "volume_mount": "kubeflow-user-workspace(pvc):/mnt",
                    "resource_memory": "8G",
                    "resource_cpu": "4",
                    "resource_gpu": "0",
                    "namespace": "jupyter"
                },indent=4,ensure_ascii=False)
            ))
            db.session.commit()
            print('添加在线构建镜像成功')
    except Exception as e:
        print(e)
        # traceback.print_exc()