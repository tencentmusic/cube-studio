import re

from flask_appbuilder.baseviews import expose_api

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.model_team import Project, Project_User
from wtforms import SelectField, StringField
from myapp.utils import core
from myapp import appbuilder, conf
from markupsafe import Markup
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from flask_appbuilder.fieldwidgets import Select2Widget, BS3TextFieldWidget
from myapp.exceptions import MyappException
from myapp import db, security_manager
from myapp.forms import MyBS3TextFieldWidget, MyBS3TextAreaFieldWidget, MySelect2Widget
from wtforms.validators import DataRequired, Regexp, Length, regexp
from flask import (
    flash,
    g
)
import pysnooper
from .base import (
    get_user_roles,
    MyappFilter,
    MyappModelView,
)
from .baseApi import (
    MyappModelRestApi
)
import json
from myapp.utils.py.py_k8s import K8s
from flask_appbuilder import CompactCRUDMixin, expose

# # 获取某类project分组
# class Project_users_Filter(MyappFilter):
#     # @pysnooper.snoop()
#     def apply(self, query, value):
#         # user_roles = [role.name.lower() for role in list(get_user_roles())]
#         # if "admin" in user_roles:
#         #     return query.filter(Project.type == value).order_by(Project.id.desc())
#         return query.filter(self.model.field == value)

# 自己是创建者的才显示,id排序显示
class Creator_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query.order_by(self.model.id.desc())

        return query.filter(self.model.created_by_fk == g.user.id).order_by(self.model.id.desc())

# 获取自己参加的某类project分组
class Project_Join_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, value):
        if g.user.is_admin():
            return query.filter(self.model.type == value).order_by(self.model.id.desc())
        join_projects_id = security_manager.get_join_projects_id(db.session)
        return query.filter(self.model.id.in_(join_projects_id)).filter(self.model.type==value).order_by(self.model.id.desc())

# table show界面下的
class Project_User_ModelView_Base():
    label_title = _('组成员')
    datamodel = SQLAInterface(Project_User)
    add_columns = ['project', 'user', 'role']
    edit_columns = add_columns
    list_columns = ['user', 'role']

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    # base_filters = [["id", Project_users_Filter, __('org')]]

    add_form_extra_fields = {
        "project": QuerySelectField(
            _('项目组'),
            query_factory=lambda: db.session.query(Project),
            allow_blank=True,
            widget=Select2Widget(extra_classes="readonly"),
            description= _('只有creator可以添加修改组成员，可以添加多个creator')
        ),
        "role": SelectField(
            _('成员角色'),
            widget=Select2Widget(),
            default='dev',
            choices=[[x, x] for x in ['dev', 'ops', 'creator']],
            description= _('只有creator可以添加修改组成员，可以添加多个creator'),
            validators=[DataRequired()]
        )
    }
    edit_form_extra_fields = add_form_extra_fields

    def pre_add_req(self,req_json,*args,**kwargs):

        if g.user and g.user.is_admin():
            return req_json
        creators = db.session().query(Project_User).filter_by(project_id=req_json.get('project')).filter_by(role='creator').all()
        creators = [creator.user.username for creator in creators]

        if g.user.username not in creators:
            raise MyappException('just creator can add/edit user')
    pre_update_req = pre_add_req

    # 校验是否有权限需改组内用户
    def check_edit_permission(self,item):
        if g.user.is_admin():
            return True
        creators = db.session().query(Project_User).filter_by(project_id=item.project_id).filter_by(role='creator').all()
        creators = [creator.user.username for creator in creators]
        if g.user.username in creators:
            return True
        return False

    check_delete_permission=check_edit_permission

    pre_update_req=pre_add_req



class Project_User_ModelView_Api(Project_User_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Project_User)
    route_base = '/project_user_modelview/api'
    # add_columns = ['user', 'role']
    add_columns = ['project', 'user', 'role']
    edit_columns = add_columns


appbuilder.add_api(Project_User_ModelView_Api)


# 获取某类project分组
class Project_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, value):
        # user_roles = [role.name.lower() for role in list(get_user_roles())]
        # if "admin" in user_roles:
        #     return query.filter(Project.type == value).order_by(Project.id.desc())
        return query.filter(self.model.type == value).order_by(self.model.id.desc())


# query joined project
# @pysnooper.snoop()
def filter_join_org_project():
    query = db.session.query(Project)
    if g.user.is_admin():
        return query.filter(Project.type == 'org').order_by(Project.id.desc())

    my_user_id = g.user.get_id() if g.user else 0
    owner_ids_query = db.session.query(Project_User.project_id).filter(Project_User.user_id == my_user_id)

    return query.filter(Project.id.in_(owner_ids_query)).filter(Project.type == 'org').order_by(Project.id.desc())


class Project_ModelView_Base():

    # @pysnooper.snoop()
    def pre_update(self, item):
        if self.pre_add:
            self.pre_add(item)

        if not item.type:
            item.type = self.project_type
        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand), indent=4, ensure_ascii=False)

    def check_edit_permission(self,item):
        if not g.user.is_admin() and g.user.username not in item.get_creators():
            return False
        return True

    def check_delete_permission(self,item):
        if not g.user.is_admin():
            return False
        return True

    # @pysnooper.snoop()
    def post_list(self, items):
        return core.sort_expand_index(items)


class Project_ModelView_job_template_Api(Project_ModelView_Base, MyappModelRestApi):
    route_base = '/project_modelview/job_template/api'
    datamodel = SQLAInterface(Project)
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    order_columns = ['id']
    cols_width = {
        "name": {"type": "ellip1", "width": 100},
        "describe": {"type": "ellip1", "width": 150},
        "user": {"type": "ellip2", "width": 700},
        "job_template": {"type": "ellip2", "width": 700},
        "type": {"type": "ellip1", "width": 100},
    }
    add_columns = ['name', 'describe', 'expand']
    search_columns=["name"]
    edit_columns = add_columns

    project_type = 'job-template'
    base_filters = [["id", Project_Filter, project_type]]
    list_columns = ['name','job_template', 'type']
    label_title = _('模板分组')
    edit_form_extra_fields = {
        'type': StringField(
            _('模板分组'),
            description='',
            widget=MyBS3TextFieldWidget(value=project_type, readonly=1),
            default=project_type,
        ),
        'expand': StringField(
            _('扩展'),
            description= _('扩展参数。示例参数：<br>"index": 0   表示在pipeline编排中的模板列表的排序位置'),
            widget=MyBS3TextAreaFieldWidget(is_json=True),
            default='{}',
        )
    }
    add_form_extra_fields = edit_form_extra_fields

    def post_add(self, item):
        if not item.type:
            item.type = self.project_type
        db.session.commit()


    # 配置只有管理员可以添加和删除
    def add_more_info(self,response, **kwargs):
        if g.user.is_admin():
            response['permissions']=['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
        else:
            response['permissions'] = ['can_list', 'can_show']

appbuilder.add_api(Project_ModelView_job_template_Api)


class Project_ModelView_org_Api(Project_ModelView_Base, MyappModelRestApi):
    route_base = '/project_modelview/org/api'
    datamodel = SQLAInterface(Project)
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    order_columns = ['id']
    cols_width = {
        "name": {"type": "ellip1", "width": 100},
        "describe": {"type": "ellip1", "width": 150},
        "user": {"type": "ellip2", "width": 700},
        "project_user":{"type": "ellip2", "width": 700},
        "type": {"type": "ellip1", "width": 100},
    }
    add_columns = ['name', 'describe', 'expand']
    search_columns=["name"]
    edit_columns = add_columns
    project_type = 'org'
    base_filters = [["id", Project_Filter, project_type]]
    spec_label_columns = {
        "cluster_name": _('集群'),
        "org":_("资源组")
    }
    list_columns = ['name', 'project_user','cluster_name','org']
    related_views = [Project_User_ModelView_Api, ]
    label_title = _('项目分组')
    edit_form_extra_fields = {
        'type': StringField(
            _('项目分组'),
            description='',
            widget=MyBS3TextFieldWidget(value=project_type, readonly=1),
            default=project_type,
        ),
        'expand': StringField(
            _('扩展'),
            description= _('扩展参数。示例参数：<br>"cluster": "dev"<br>"org": "public"<br>"volume_mount": "kubeflow-user-workspace(pvc):/mnt/;/data/k8s/../group1(hostpath):/group1"<br>"SERVICE_EXTERNAL_IP":"xx.内网.xx.xx|xx.公网.xx.xx"'),
            widget=MyBS3TextAreaFieldWidget(),
            default=json.dumps({"cluster": "dev", "org" : "public"}, indent=4, ensure_ascii=False),
        )
    }
    add_form_extra_fields = edit_form_extra_fields

    expand_columns = {
        "expand": {
            "cluster": SelectField(
                label= _('集群'),
                widget=Select2Widget(),
                default='dev',
                description= _('使用该项目组的所有任务部署到的目的集群'),
                choices=[[x, x] for x in list(conf.get('CLUSTERS', {"dev": {}}).keys())],
                validators=[DataRequired()]
            ),
            'volume_mount': StringField(
                label= _('挂载'),
                default='kubeflow-user-workspace(pvc):/mnt/',
                description= _('<span style="word-break: break-all;">使用该项目组新增的所有pod会自动添加该挂载，格式:<br>$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,$storage_name(storage):/$container_path3 <br>注意pvc会自动挂载对应目录下的个人username子目录。示例：<br>kubeflow-user-workspace(pvc):/mnt/,/data/k8s/kubeflow/group1(hostpath):/group1,nfs-test(storage):/nfs</span>'),
                widget=BS3TextFieldWidget(),
                validators=[Regexp('^[\x00-\x7F]*$')]
            ),
            'SERVICE_EXTERNAL_IP': StringField(
                label = _('服务代理ip'),
                default='',
                description = _("服务的代理ip，xx.内网.xx.xx|xx.公网.xx.xx"),
                widget=BS3TextFieldWidget()
            ),
            "org": SelectField(
                label = _('资源组'),
                widget = MySelect2Widget(can_input=True),
                default='public',
                choices=[[x, x] for x in ['public']],
                description = _('使用该项目组的所有任务部署到的目的资源组，通过机器label org=xx决定'),
                validators=[DataRequired()]
            )
        }
    }

    # 根据命名空间的字符串配置获取真实命名空间
    def get_project_namespaces(self,namespace):
        
        real_namespace = {
            "NOTEBOOK_NAMESPACE": conf.get('NOTEBOOK_NAMESPACE', 'jupyter'),
            "PIPELINE_NAMESPACE": conf.get('PIPELINE_NAMESPACE', 'pipeline'),
            "SERVICE_NAMESPACE": conf.get('SERVICE_NAMESPACE', 'service'),
            "AUTOML_NAMESPACE": conf.get('AUTOML_NAMESPACE', 'automl'),
        }
        
        return real_namespace
    # @pysnooper.snoop(watch_explode=('req_json'))
    def pre_add_req(self, req_json, *args, **kwargs):

        volumns = req_json.get('volume_mount', '') or ''
        quotas = req_json.get('quota', '') or ''
        quotas = [x.strip().replace(' ', '') for x in quotas.strip().split(';') if x.strip()]
        cluster = req_json.get('cluster', conf.get('ENVIRONMENT', ''))
        config_path = conf.get('CLUSTERS', {}).get(cluster, {}).get('KUBECONFIG', '')
        k8s_client = K8s(config_path)

        volumns = [x.strip().replace(' ', '') for x in re.split(',|;', volumns.replace('，',','))]
        new_volumns = []
        for index, volumn in enumerate(volumns):
            if '(pvc)' in volumn or '(pvc-share)' in volumn:
                pvc_name = volumn.split(':')[0].strip().replace('(pvc)', '').replace('(pvc-share)','').strip()
                for namespace in list(set(self.get_project_namespaces(req_json.get('namespace','')).values())):
                    pvc = k8s_client.get_pvc(name=pvc_name, namespace=namespace)
                    if not pvc:
                        raise Exception(f'pvc {pvc_name} not exist in {namespace} namespace')
            if '(storage)' in volumn:
                from myapp.models.model_storage import Storage
                storage_name = volumn.split(':')[0].strip().replace('(storage)', '').strip()
                storage = db.session.query(Storage).filter_by(name=storage_name).first()
                if not storage:
                    raise Exception(f'storage {storage_name} not exist')
            if volumn.strip() and '(pvc)' not in volumn and '(pvc-share)' not in volumn and '(hostpath)' not in volumn and '(storage)' not in volumn and '(configmap)' not in volumn and '(memory)' not in volumn:
                if ':' in volumn and '(' not in volumn:
                    volumn = volumn.split(':')[0] + "(hostpath):" + volumn.split(':')[1]
                else:
                    raise Exception(f'find invalid volumn config {volumn}, must belong (pvc)(pvc-share)(hostpath)(storage)(configmap)(memory)')
            new_volumns.append(volumn)
        for quota in quotas:
            if len(quota.replace('，',',').split(',')) != 4:
                raise Exception(f'quota {quota} config error')
        # 做一些书写修正
        volumns = ';'.join(new_volumns)
        quotas = ';'.join(quotas)
        req_json['volume_mount']=volumns
        req_json['quotas'] = quotas
        return req_json

    pre_update_req = pre_add_req

    # add project user
    def post_add(self, item):
        if not item.type:
            item.type = self.project_type
        creator = Project_User(role='creator', user=g.user, project=item)
        db.session.add(creator)
        db.session.commit()


    # @pysnooper.snoop(watch_explode=('response',))
    def add_more_info(self, response, **kwargs):
        if g.user.is_admin():
            response['permissions']=['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
        else:
            response['permissions'] = ['can_list', 'can_show']

appbuilder.add_api(Project_ModelView_org_Api)

class Project_ModelView_Api(Project_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Project)
    route_base = '/project_modelview/api'
    related_views = [Project_User_ModelView_Api, ]




appbuilder.add_api(Project_ModelView_Api)
