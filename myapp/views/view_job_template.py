from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
from wtforms.validators import DataRequired, Length, Regexp
import pysnooper
from sqlalchemy.exc import InvalidRequestError

from myapp.models.model_job import Job_Template
from flask_appbuilder.actions import action
from jinja2 import Environment, BaseLoader, DebugUndefined
from myapp.utils import core
from myapp import app, appbuilder, db

from wtforms import BooleanField, StringField, SelectField

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MyCodeArea
import logging
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
from myapp.views.view_team import Project_Filter,Creator_Filter
import datetime, time, json

conf = app.config


# 开发者能看到所有模板，用户只能看到release的模板
class Job_Tempalte_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query.order_by(self.model.id.desc())

        # join_projects_id = security_manager.get_join_projects_id(db.session)
        # logging.info(join_projects_id)
        return query.filter(self.model.version == 'Release').order_by(self.model.id.desc())


class Job_Template_ModelView_Base():
    datamodel = SQLAInterface(Job_Template)
    label_title = _('任务模板')
    check_redirect_list_url = conf.get('MODEL_URLS', {}).get('job_template', '')

    list_columns = ['project', 'name_title', 'version', 'creator', 'modified']
    spec_label_columns = {
        "project": _("功能分类")
    }
    cols_width = {
        "name_title": {"type": "ellip2", "width": 300},
        "name": {"type": "ellip2", "width": 400},
        "version": {"type": "ellip2", "width": 100},
        "modified": {"type": "ellip2", "width": 200},
    }
    show_columns = ['project', 'name', 'version', 'describe', 'images_url', 'workdir', 'entrypoint', 'args_html',
                    'demo_html', 'env', 'hostAliases', 'privileged', 'expand_html']
    add_columns = ['project', 'images', 'name', 'version', 'describe', 'workdir', 'entrypoint', 'volume_mount',
                   'job_args_definition', 'args', 'env', 'hostAliases', 'privileged', 'accounts', 'demo', 'expand']
    edit_columns = add_columns

    base_filters = [["id", Job_Tempalte_Filter, lambda: []]]
    base_order = ('id', 'desc')
    order_columns = ['id']
    add_form_query_rel_fields = {
        "images": [["name", Creator_Filter, None]],
        "project": [["name", Project_Filter, 'job-template']],
    }
    version_list = [[version, version] for version in ['Alpha', 'Release']]
    edit_form_query_rel_fields = add_form_query_rel_fields
    add_form_extra_fields = {
        "name": StringField(
            _('名称'),
            description= _('英文名(小写字母、数字、- 组成)，最长50个字符'),
            default='',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]
        ),
        "describe": StringField(
            _("描述"),
            description= _("模板的描述将直接显示在pipeline编排界面"),
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "version": SelectField(
            _('版本'),
            description= _("job模板的版本，release版本的模板才能被所有用户看到"),
            default='',
            widget=Select2Widget(),
            choices=version_list
        ),
        "volume_mount": StringField(
            _('挂载'),
            default='',
            description= _('使用该模板的task，会在添加时，自动添加该挂载。<br>外部挂载，格式示例:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,4G(memory):/dev/shm,注意pvc会自动挂载对应目录下的个人username子目录'),
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "workdir": StringField(
            _('工作目录'),
            description= _('工作目录，不填写将直接使用镜像默认的工作目录'),
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "entrypoint": StringField(
            _('启动命令'),
            description= _('镜像的入口命令，直接写成单行字符串，例如python xx.py，无需添加[]'),
            default='',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "job_args_definition": StringField(
            _('参数定义'),
            description= _('使用job模板参数的标准填写方式'),
            widget=MyCodeArea(code=core.job_template_args_definition()),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "env": StringField(
            _('环境变量'),
            default='',
            description= _('使用模板的task自动添加的环境变量，支持模板变量。<br>书写格式:每行一个环境变量env_key=env_value'),
            widget=MyBS3TextAreaFieldWidget(rows=3),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "hostAliases": StringField(
            _('域名解析'),
            default='',
            description= _('添加到容器内的host映射。<br>书写格式:每行一个dns解析记录，ip host1 host2，<br>示例：1.1.1.1 example1.oa.com example2.oa.com'),
            widget=MyBS3TextAreaFieldWidget(rows=3),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "demo": StringField(
            _('demo'),
            description= _('填写demo'),
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "accounts": StringField(
            _('k8s账号'),
            default='',
            description= _('k8s的ServiceAccount，在此类任务运行时会自动挂载此账号，多用于模板用于k8s pod/cr时使用'),
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
            validators=[]
        ),
        "privileged": BooleanField(
            _('超级权限'),
            description= _('是否启动超级权限')
        ),
        "expand": StringField(
            _('扩展'),
            default=json.dumps({"index": 0, "help_url": conf.get('DOCUMENTATION_URL')}, ensure_ascii=False, indent=4),
            description= _('json格式的扩展字段，支持<br> "index":"$模板展示顺序号"，<br>"help_url":"$帮助文档地址"，<br>"HostNetwork":true 启动主机端口监听'),
            widget=MyBS3TextAreaFieldWidget(rows=3),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        )
    }
    edit_form_extra_fields = add_form_extra_fields

    def set_columns(self,job_template=None):
        args_field = StringField(
            _('参数'),
            default=json.dumps({
                __("参数分组1"): {
                    "--attr1": {
                        "type": "str",
                        "label": __("参数1"),
                        "default": "value1",
                        "describe": __("这里是这个参数的描述和备注"),
                    }
                }
            }, indent=4, ensure_ascii=False),
            description= _('json格式，此类task使用时需要填写的参数，示例：')+'<br><pre><code>%s</code></pre>' % core.job_template_args_definition(),
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
            validators=[DataRequired()]
        )

        self.edit_form_extra_fields['args'] = args_field
        self.add_form_extra_fields['args'] = args_field

    pre_add_web = set_columns
    pre_update_web = set_columns

    # 校验是否是json
    # @pysnooper.snoop(watch_explode=('job_args'))
    def pre_add(self, item):
        if not item.env:
            item.env = ''
        envs = item.env.strip().split('\n')
        envs = [env.strip() for env in envs if env.strip() and '=' in env]
        item.env = '\n'.join(envs)
        if not item.args:
            item.args = '{}'
        item.args = core.validate_job_args(item)

        if not item.expand or not item.expand.strip():
            item.expand = '{}'
        core.validate_json(item.expand)
        item.expand = json.dumps(json.loads(item.expand), indent=4, ensure_ascii=False)

        if not item.demo or not item.demo.strip():
            item.demo = '{}'

        core.validate_json(item.demo)

        if item.hostAliases:
            # if not item.images.entrypoint:
            #     raise MyappException('images entrypoint not exist')
            all_host = {}
            all_rows = re.split('\r|\n', item.hostAliases)
            all_rows = [all_row.strip() for all_row in all_rows if all_row.strip()]
            for row in all_rows:
                hosts = row.split(' ')
                hosts = [host for host in hosts if host]
                if len(hosts) > 1:
                    if hosts[0] in all_host:
                        all_host[hosts[0]] = all_host[hosts[0]] + hosts[1:]
                    else:
                        all_host[hosts[0]] = hosts[1:]

            hostAliases = ''
            for ip in all_host:
                hostAliases += ip + " " + " ".join(all_host[ip])
                hostAliases += '\n'
            item.hostAliases = hostAliases.strip()

        task_args = json.loads(item.demo)
        job_args = json.loads(item.args)
        item.demo = json.dumps(core.validate_task_args(task_args, job_args), indent=4, ensure_ascii=False)

    # 检测是否具有编辑权限，只有creator和admin可以编辑
    def check_edit_permission(self, item):
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return True
        if g.user and g.user.username and hasattr(item, 'created_by'):
            if g.user.username == item.created_by.username:
                return True
        flash('just creator can edit/delete ', 'warning')
        return False

    def pre_update(self, item):
        self.pre_add(item)

    # @pysnooper.snoop()
    def post_list(self,items):
        def sort_expand_index(items):
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
        return sort_expand_index(items)



    @action("copy", "复制", confirmation= '复制所选记录?', icon="fa-copy",multiple=True, single=False)
    def copy(self, job_templates):
        if not isinstance(job_templates, list):
            job_templates = [job_templates]
        try:
            for job_template in job_templates:
                new_job_template = job_template.clone()
                new_job_template.name = new_job_template.name + "_copy_" + uuid.uuid4().hex[:4]
                new_job_template.created_on = datetime.datetime.now()
                new_job_template.changed_on = datetime.datetime.now()
                db.session.add(new_job_template)
                db.session.commit()
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            raise e
        return redirect(request.referrer)

# 添加api
class Job_Template_ModelView_Api(Job_Template_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Job_Template)
    page_size = 1000
    route_base = '/job_template_modelview/api'
    # add_columns = ['project', 'images', 'name', 'version', 'describe', 'args', 'env','hostAliases', 'privileged','accounts', 'demo','expand']
    add_columns = ['project', 'images', 'name', 'version', 'describe', 'workdir', 'entrypoint', 'volume_mount', 'args',
                   'env', 'hostAliases', 'privileged', 'accounts', 'expand']
    edit_columns = add_columns
    # list_columns = ['project','name','version','creator','modified']
    list_columns = ['project', 'name', 'version', 'describe', 'images', 'workdir', 'entrypoint', 'args', 'demo', 'env',
                    'hostAliases', 'privileged', 'accounts', 'created_by', 'changed_by', 'created_on', 'changed_on',
                    'expand']
    show_columns = ['project', 'name', 'version', 'describe', 'images', 'workdir', 'entrypoint', 'args', 'demo', 'env',
                    'hostAliases', 'privileged', 'expand']


appbuilder.add_api(Job_Template_ModelView_Api)


class Job_Template_fab_ModelView_Api(Job_Template_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Job_Template)
    route_base = '/job_template_fab_modelview/api'
    # add_columns = ['project', 'images', 'name', 'version', 'describe', 'args', 'env','hostAliases', 'privileged','accounts', 'demo','expand']
    add_columns = ['project', 'images', 'name', 'version', 'describe', 'workdir', 'entrypoint', 'volume_mount', 'args',
                   'env', 'hostAliases', 'privileged', 'accounts', 'expand']
    page_size = 1000
    edit_columns = add_columns
    list_columns = ['project', 'name', 'version', 'creator', 'modified']
    show_columns = ['project', 'images', 'name', 'version', 'describe', 'workdir', 'entrypoint', 'volume_mount', 'args',
                    'env', 'hostAliases', 'privileged', 'accounts', 'expand']

appbuilder.add_api(Job_Template_fab_ModelView_Api)
