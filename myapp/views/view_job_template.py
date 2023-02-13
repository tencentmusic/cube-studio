from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
from wtforms.validators import DataRequired, Length, Regexp

from sqlalchemy.exc import InvalidRequestError

from myapp.models.model_job import Job_Template
from flask_appbuilder.actions import action
from jinja2 import Environment, BaseLoader, DebugUndefined
from myapp.utils import core
from myapp import app, appbuilder,db

from wtforms import BooleanField, StringField, SelectField

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MyCodeArea

import re

from .baseApi import (
    MyappModelRestApi
)
from flask import (
    flash,
    g,
    Markup,
    make_response,
    redirect,
    request
)

from .base import (
    get_user_roles,
    MyappFilter,
)
from flask_appbuilder import expose
from myapp.views.view_images import Images_Filter
from myapp.views.view_team import Project_Filter
import datetime,time,json
conf = app.config
logging = app.logger




# 开发者能看到所有模板，用户只能看到release的模板
class Job_Tempalte_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        # join_projects_id = security_manager.get_join_projects_id(db.session)
        # logging.info(join_projects_id)
        return query.filter(self.model.version=='Release')


class Job_Template_ModelView_Base():
    datamodel = SQLAInterface(Job_Template)
    label_title='任务模板'
    check_redirect_list_url = conf.get('MODEL_URLS',{}).get('job_template','')

    list_columns = ['project','name_title','version','creator','modified']
    cols_width = {
        "name_title":{"type": "ellip2", "width": 300},
        "name": {"type": "ellip2", "width": 400},
        "version": {"type": "ellip2", "width": 100},
        "modified": {"type": "ellip2", "width": 200},
    }
    show_columns = ['project','name','version','describe','images_url','workdir','entrypoint','args_html','demo_html','env','hostAliases','privileged','expand_html']
    add_columns = ['project','images','name','version','describe','workdir','entrypoint','volume_mount','job_args_definition','args','env','hostAliases','privileged','accounts','demo','expand']
    edit_columns = add_columns

    base_filters = [["id", Job_Tempalte_Filter, lambda: []]]
    base_order = ('created_on', 'desc')
    order_columns = ['id']
    add_form_query_rel_fields = {
        "images": [["name", Images_Filter, None]],
        "project": [["name", Project_Filter, 'job-template']],
    }
    version_list = [[version,version] for version in ['Alpha','Release']]
    edit_form_query_rel_fields = add_form_query_rel_fields
    add_form_extra_fields = {
        "name": StringField(
            _(datamodel.obj.lab('name')),
            description='英文名(小写字母、数字、- 组成)，最长50个字符',
            default='',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]
        ),
        "describe": StringField(
            _(datamodel.obj.lab('describe')),
            description="模板的描述将直接显示在pipeline编排界面",
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "version": SelectField(
            _(datamodel.obj.lab('version')),
            description="job模板的版本，release版本的模板才能被所有用户看到",
            default='',
            widget=Select2Widget(),
            choices=version_list
        ),
        "volume_mount": StringField(
            _(datamodel.obj.lab('volume_mount')),
            default='',
            description='使用该模板的task，会在添加时，自动添加该挂载。<br>外部挂载，格式示例:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,4G(memory):/dev/shm,注意pvc会自动挂载对应目录下的个人rtx子目录',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "workdir": StringField(
            _(datamodel.obj.lab('workdir')),
            description='工作目录，不填写将直接使用镜像默认的工作目录',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "entrypoint": StringField(
            _(datamodel.obj.lab('entrypoint')),
            description='镜像的入口命令，直接写成单行字符串，例如python xx.py，无需添加[]',
            default='',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "job_args_definition": StringField(
            _(datamodel.obj.lab('job_args_definition')),
            description='使用job模板参数的标准填写方式',
            widget=MyCodeArea(code=core.job_template_args_definition()),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "args": StringField(
            _(datamodel.obj.lab('args')),
            default=json.dumps({
                "参数分组1":{
                   "--attr1":{
                    "type":"str",
                    "label":"参数1",
                    "default":"value1",
                    "describe":"这里是这个参数的描述和备注",
                  }
                }
            },indent=4,ensure_ascii=False),
            description=Markup('json格式，此类task使用时需要填写的参数，示例：<br><pre><code>%s</code></pre>'%core.job_template_args_definition()),
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
            validators=[DataRequired()]
        ),
        "env": StringField(
            _(datamodel.obj.lab('env')),
            default='',
            description='使用模板的task自动添加的环境变量，支持模板变量。<br>书写格式:每行一个环境变量env_key=env_value',
            widget=MyBS3TextAreaFieldWidget(rows=3),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "hostAliases": StringField(
            _(datamodel.obj.lab('hostAliases')),
            default='',
            description='添加到容器内的host映射。<br>书写格式:每行一个dns解析记录，ip host1 host2，<br>示例：1.1.1.1 example1.oa.com example2.oa.com',
            widget=MyBS3TextAreaFieldWidget(rows=3),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "demo": StringField(
            _(datamodel.obj.lab('demo')),
            description='填写demo',
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "accounts": StringField(
            _(datamodel.obj.lab('accounts')),
            default='',
            description='k8s的ServiceAccount，在此类任务运行时会自动挂载此账号，多用于模板用于k8s pod/cr时使用',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
            validators=[]
        ),
        "privileged":BooleanField(
            _(datamodel.obj.lab('privileged')),
            description='是否启动超级权限'
        ),
        "expand": StringField(
            _(datamodel.obj.lab('expand')),
            default=json.dumps({"index":0,"help_url":"https://github.com/tencentmusic/cube-studio"},ensure_ascii=False,indent=4),
            description='json格式的扩展字段，支持 "index":"$模板展示顺序号"，"help_url":"$帮助文档地址"',
            widget=MyBS3TextAreaFieldWidget(rows=3),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
    }
    edit_form_extra_fields = add_form_extra_fields

    # 校验是否是json
    # @pysnooper.snoop(watch_explode=('job_args'))
    def pre_add(self, item):
        if not item.env:
            item.env=''
        envs = item.env.strip().split('\n')
        envs = [env.strip() for env in envs if env.strip() and '=' in env]
        item.env = '\n'.join(envs)
        if not item.args:
            item.args = '{}'
        item.args = core.validate_job_args(item)

        if not item.expand or not item.expand.strip():
            item.expand='{}'
        core.validate_json(item.expand)
        item.expand = json.dumps(json.loads(item.expand), indent=4, ensure_ascii=False)

        if not item.demo or not item.demo.strip():
            item.demo='{}'

        core.validate_json(item.demo)

        if item.hostAliases:
            # if not item.images.entrypoint:
            #     raise MyappException('images entrypoint not exist')
            all_host = {}
            all_rows = re.split('\r|\n',item.hostAliases)
            all_rows = [all_row.strip() for all_row in all_rows if all_row.strip()]
            for row in all_rows:
                hosts = row.split(' ')
                hosts = [host for host in hosts if host]
                if len(hosts) > 1:
                    if hosts[0] in all_host:
                        all_host[hosts[0]]=all_host[hosts[0]]+hosts[1:]
                    else:
                        all_host[hosts[0]] = hosts[1:]

            hostAliases=''
            for ip in all_host:
                hostAliases+=ip+" "+" ".join(all_host[ip])
                hostAliases+='\n'
            item.hostAliases = hostAliases.strip()

        task_args = json.loads(item.demo)
        job_args = json.loads(item.args)
        item.demo = json.dumps(core.validate_task_args(task_args, job_args),indent=4, ensure_ascii=False)

    # 检测是否具有编辑权限，只有creator和admin可以编辑
    def check_edit_permission(self, item):
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return True
        if g.user and g.user.username and hasattr(item,'created_by'):
            if g.user.username==item.created_by.username:
                return True
        flash('just creator can edit/delete ', 'warning')
        return False


    def pre_update(self, item):
        self.pre_add(item)

    # @pysnooper.snoop()
    def post_list(self,items):
        def sort_expand_index(items, dbsession):
            all = {
                0: []
            }
            for item in items:
                try:
                    if item.expand:
                        index = float(json.loads(item.expand).get('index', 0))+float(json.loads(item.project.expand).get('index', 0))*1000
                        if index:
                            if index in all:
                                all[index].append(item)
                            else:
                                all[index] = [item]
                        else:
                            all[0].append(item)
                    else:
                        all[0].append(item)
                except Exception as e:
                    print(e)
            back = []
            for index in sorted(all):
                back.extend(all[index])
                # 当有小数的时候自动转正
                # if float(index)!=int(index):
                #     pass
            return back
        return sort_expand_index(items,db.session)



    @expose("/run", methods=["POST"])
    def run(self):
        request_data = request.json
        job_template_id=request_data.get('job_template_id','')
        job_template_name = request_data.get('job_template_name', '')
        run_id = request_data.get('run_id', '').replace('_','-')
        resource_memory = request_data.get('resource_memory', '')
        resource_cpu = request_data.get('resource_cpu', '')
        task_args = request_data.get('args', '')
        if (not job_template_id and not job_template_name) or not run_id or task_args=='':
            response = make_response("输入参数不齐全")
            response.status_code = 400
            return response

        job_template = None
        if job_template_id:
            job_template = db.session.query(Job_Template).filter_by(id=int(job_template_id)).first()
        elif job_template_name:
            job_template = db.session.query(Job_Template).filter_by(name=job_template_name).first()
        if not job_template:
            response = make_response("no job template exist")
            response.status_code = 400
            return response

        from myapp.utils.py.py_k8s import K8s

        k8s = K8s()
        namespace = conf.get('PIPELINE_NAMESPACE')
        pod_name = "venus-" + run_id.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')
        pod = k8s.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            k8s.delete_pods(namespace=namespace, pod_name=pod_name)
            time.sleep(2)
            pod=None
        # 没有历史或者没有运行态，直接创建
        if not pod:
            args=[]

            job_template_args = json.loads(job_template.args) if job_template.args else {}
            for arg_name in task_args:
                arg_type=''
                for group in job_template_args:
                    for template_arg in job_template_args[group]:
                        if template_arg==arg_name:
                            arg_type=job_template_args[group][template_arg].get('type','')
                arg_value = task_args[arg_name]
                if arg_value:
                    args.append(arg_name)
                    if arg_type=='json':
                        args.append(json.dumps(arg_value))
                    else:
                        args.append(arg_value)

            # command = ['sh', '-c','sleep 7200']
            volume_mount = 'kubeflow-cfs-workspace(pvc):/mnt,kubeflow-cfs-archives(pvc):/archives'
            env = job_template.env + "\n"
            env += 'KFJ_TASK_ID=0\n'
            env += 'KFJ_TASK_NAME=' + str('venus-'+run_id) + "\n"
            env += 'KFJ_TASK_NODE_SELECTOR=cpu=true,train=true\n'
            env += 'KFJ_TASK_VOLUME_MOUNT=' + str(volume_mount) + "\n"
            env += 'KFJ_TASK_IMAGES=' + str(job_template.images) + "\n"
            env += 'KFJ_TASK_RESOURCE_CPU=' + str(resource_cpu) + "\n"
            env += 'KFJ_TASK_RESOURCE_MEMORY=' + str(resource_memory) + "\n"
            env += 'KFJ_TASK_RESOURCE_GPU=0\n'
            env += 'KFJ_PIPELINE_ID=0\n'
            env += 'KFJ_RUN_ID=' + run_id + "\n"
            env += 'KFJ_CREATOR=' + str(g.user.username) + "\n"
            env += 'KFJ_RUNNER=' + str(g.user.username) + "\n"
            env += 'KFJ_PIPELINE_NAME=venus\n'
            env += 'KFJ_NAMESPACE=pipeline' + "\n"


            def template_str(src_str):
                rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(src_str)
                des_str = rtemplate.render(creator=g.user.username,
                                           datetime=datetime,
                                           runner=g.user.username,
                                           uuid=uuid,
                                           pipeline_id='0',
                                           pipeline_name='venus-task',
                                           cluster_name=conf.get('ENVIRONMENT')
                                           )
                return des_str

            global_envs = json.loads(template_str(json.dumps(conf.get('GLOBAL_ENV', {}), indent=4, ensure_ascii=False)))
            for global_env_key in global_envs:
                env += global_env_key + '=' + global_envs[global_env_key] + "\n"

            hostAliases=job_template.hostAliases+"\n"+conf.get('HOSTALIASES','')
            k8s.create_debug_pod(namespace,
                                 name=pod_name,
                                 labels={"app":"docker","user":g.user.username,"pod-type":"job-template"},
                                 command=None,
                                 args=args,
                                 volume_mount=volume_mount,
                                 working_dir=None,
                                 node_selector='cpu=true,train=true',
                                 resource_cpu=resource_cpu,
                                 resource_memory=resource_memory,
                                 resource_gpu=0,
                                 image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
                                 image_pull_secrets=[job_template.images.repository.hubsecret],
                                 image=job_template.images.name,
                                 hostAliases=hostAliases,
                                 env=env,
                                 privileged=job_template.privileged,
                                 accounts=job_template.accounts,
                                 username=g.user.username
                                 )

        try_num = 5
        while (try_num > 0):
            pod = k8s.get_pods(namespace=namespace, pod_name=pod_name)
            # print(pod)
            if pod:
                break
            try_num = try_num - 1
            time.sleep(2)
        if try_num == 0:
            response = make_response("启动时间过长，一分钟后重试")
            response.status_code = 400
            return response

        user_roles = [role.name.lower() for role in list(g.user.roles)]
        if "admin" in user_roles:
            pod_url = conf.get('K8S_DASHBOARD_CLUSTER') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, pod_name)
        else:
            pod_url = conf.get('K8S_DASHBOARD_PIPELINE') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, pod_name)
        print(pod_url)
        response = make_response("启动成功，日志地址: %s"%pod_url)
        response.status_code = 200
        return response




    @expose("/listen", methods=["POST"])
    def listen(self):
        request_data = request.json
        run_id = request_data.get('run_id', '').replace('_','-')
        if not run_id:
            response = make_response("输入参数不齐全")
            response.status_code = 400
            return response

        from myapp.utils.py.py_k8s import K8s
        k8s = K8s()
        namespace = conf.get('PIPELINE_NAMESPACE')
        pod_name = "venus-" + run_id.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')
        pod = k8s.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
            if type(pod['start_time'])==datetime.datetime:
                pod['start_time'] = pod['start_time'].strftime("%Y-%d-%m %H:%M:%S")
            print(pod)
            response = make_response(json.dumps(pod))
            response.status_code = 200
            return response
        else:
            response = make_response('no pod')
            response.status_code = 400
            return response



    @action(
        "copy", __("Copy Job Template"), confirmation=__('Copy Job Template'), icon="fa-copy",multiple=True, single=False
    )
    def copy(self, job_templates):
        if not isinstance(job_templates, list):
            job_templates = [job_templates]
        try:
            for job_template in job_templates:
                new_job_template = job_template.clone()
                new_job_template.name = new_job_template.name+"_copy_"+uuid.uuid4().hex[:4]
                new_job_template.created_on = datetime.datetime.now()
                new_job_template.changed_on = datetime.datetime.now()
                db.session.add(new_job_template)
                db.session.commit()
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            raise e
        return redirect(request.referrer)
#
# class Job_Template_ModelView(Job_Template_ModelView_Base,MyappModelView,DeleteMixin):
#     datamodel = SQLAInterface(Job_Template)
#
# appbuilder.add_view(Job_Template_ModelView,"任务模板",href="/job_template_modelview/list/?_flt_2_name=",icon = 'fa-flag-o',category = '训练',category_icon = 'fa-envelope')

# 添加api
class Job_Template_ModelView_Api(Job_Template_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Job_Template)
    page_size = 1000
    route_base = '/job_template_modelview/api'
    # add_columns = ['project', 'images', 'name', 'version', 'describe', 'args', 'env','hostAliases', 'privileged','accounts', 'demo','expand']
    add_columns = ['project', 'images', 'name', 'version', 'describe', 'workdir', 'entrypoint', 'volume_mount','args', 'env', 'hostAliases', 'privileged', 'accounts', 'expand']
    edit_columns = add_columns
    # list_columns = ['project','name','version','creator','modified']
    list_columns = ['project', 'name', 'version', 'describe', 'images', 'workdir', 'entrypoint', 'args', 'demo', 'env',
                    'hostAliases', 'privileged', 'accounts', 'created_by', 'changed_by', 'created_on', 'changed_on',
                    'expand']
    show_columns = ['project','name','version','describe','images','workdir','entrypoint','args','demo','env','hostAliases','privileged','expand']

appbuilder.add_api(Job_Template_ModelView_Api)

class Job_Template_fab_ModelView_Api(Job_Template_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Job_Template)
    route_base = '/job_template_fab_modelview/api'
    # add_columns = ['project', 'images', 'name', 'version', 'describe', 'args', 'env','hostAliases', 'privileged','accounts', 'demo','expand']
    add_columns = ['project', 'images', 'name', 'version', 'describe', 'workdir', 'entrypoint', 'volume_mount','args', 'env', 'hostAliases', 'privileged', 'accounts', 'expand']
    page_size = 1000
    edit_columns = add_columns
    list_columns = ['project','name','version','creator','modified']
    show_columns = ['project', 'images', 'name', 'version', 'describe', 'workdir', 'entrypoint', 'volume_mount','args', 'env', 'hostAliases', 'privileged', 'accounts', 'expand']


appbuilder.add_api(Job_Template_fab_ModelView_Api)


