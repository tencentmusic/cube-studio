from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import lazy_gettext as _
from myapp.models.model_team import Project, Project_User
from wtforms import SelectField, StringField
from myapp.utils import core
from myapp import appbuilder, conf
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from flask_appbuilder.fieldwidgets import Select2Widget, BS3TextFieldWidget
from myapp.exceptions import MyappException
from myapp import db, security_manager
from myapp.forms import MyBS3TextFieldWidget, MyBS3TextAreaFieldWidget
from wtforms.validators import DataRequired
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
from flask_appbuilder import CompactCRUDMixin


# table show界面下的
class Project_User_ModelView_Base():
    label_title = '组成员'
    datamodel = SQLAInterface(Project_User)
    add_columns = ['project', 'user', 'role']
    edit_columns = add_columns
    list_columns = ['user', 'role']

    add_form_extra_fields = {
        "project": QuerySelectField(
            "项目组",
            query_factory=lambda: db.session.query(Project),
            allow_blank=True,
            widget=Select2Widget(extra_classes="readonly"),
            description='只有creator可以添加修改组成员，可以添加多个creator'
        ),
        "role": SelectField(
            "成员角色",
            widget=Select2Widget(),
            default='dev',
            choices=[[x, x] for x in ['dev', 'ops', 'creator']],
            description='只有creator可以添加修改组成员，可以添加多个creator',
            validators=[DataRequired()]
        )
    }
    edit_form_extra_fields = add_form_extra_fields

    # @pysnooper.snoop()
    def pre_add_req(self,req_json):
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return req_json
        creators = db.session().query(Project_User).filter_by(project_id=req_json.get('project')).all()
        print(creators)
        if g.user.username not in creators:
            raise MyappException('just creator can add/edit user')

    pre_update_req=pre_add_req


class Project_User_ModelView(Project_User_ModelView_Base, CompactCRUDMixin, MyappModelView):
    datamodel = SQLAInterface(Project_User)


appbuilder.add_view_no_menu(Project_User_ModelView)


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


# 获取自己参加的某类project分组
class Project_Join_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, value):
        if g.user.is_admin():
            return query.filter(self.model.type == value).order_by(self.model.id.desc())
        join_projects_id = security_manager.get_join_projects_id(db.session)
        return query.filter(self.model.id.in_(join_projects_id)).filter(self.model.type==value).order_by(self.model.id.desc())



# query joined project
def filter_join_org_project():
    query = db.session.query(Project)
    # user_roles = [role.name.lower() for role in list(get_user_roles())]
    # if "admin" in user_roles:
    if g.user.is_admin():
        return query.filter(Project.type == 'org').order_by(Project.id.desc())

    my_user_id = g.user.get_id() if g.user else 0
    owner_ids_query = db.session.query(Project_User.project_id).filter(Project_User.user_id == my_user_id)

    return query.filter(Project.id.in_(owner_ids_query)).filter(Project.type == 'org').order_by(Project.id.desc())


class Project_ModelView_Base():
    label_title = '项目组'
    datamodel = SQLAInterface(Project)
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    list_columns = ['name', 'user', 'type']
    cols_width = {
        "name": {"type": "ellip2", "width": 200},
        "user": {"type": "ellip3", "width": 700},
        "type": {"type": "ellip2", "width": 200},
    }
    related_views = [Project_User_ModelView, ]
    add_columns = ['name', 'describe', 'expand'] # 'cluster','volume_mount','service_external_ip',
    edit_columns = add_columns
    project_type = 'org'


    add_form_extra_fields = {
        'name': StringField(
            label=_(datamodel.obj.lab('name')),
            default='',
            description='项目名称',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        'describe': StringField(
            label=_(datamodel.obj.lab('describe')),
            default='',
            description='项目描述',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

    }
    edit_form_extra_fields = add_form_extra_fields


    # @pysnooper.snoop()
    def pre_add_web(self):
        self.edit_form_extra_fields['type'] = StringField(
            _(self.datamodel.obj.lab('type')),
            description="项目分组",
            widget=MyBS3TextFieldWidget(value=self.project_type, readonly=1),
            default=self.project_type,
        )
        self.add_form_extra_fields = self.edit_form_extra_fields

    def pre_update(self, item):
        if not item.type:
            item.type = self.project_type
        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand), indent=4, ensure_ascii=False)
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return
        if not g.user.username in item.get_creators():
            raise MyappException('just creator can add/edit')


    # before update, check permission
    def pre_update_web(self, item):
        self.pre_add_web()
        self.check_item_permissions(item)
        if not self.user_permissions['edit']:
            flash('just creator can add/edit user', 'warning')
            raise MyappException('just creator can add/edit user')

    # check permission
    def check_item_permissions(self, item):
        if g.user.is_admin() or g.user.username in item.get_creators():
            self.user_permissions = {
                "add": True,
                "edit": True,
                "delete": True,
                "show": True
            }
        else:
            self.user_permissions = {
                "add": True,
                "edit": False,
                "delete": False,
                "show": True
            }

    # add project user
    def post_add(self, item):
        if not item.type:
            item.type = self.project_type
        creator = Project_User(role='creator', user=g.user, project=item)
        db.session.add(creator)
        db.session.commit()

    # @pysnooper.snoop()
    def post_list(self, items):
        return core.sort_expand_index(items)



class Project_ModelView_job_template_Api(Project_ModelView_Base, MyappModelRestApi):
    route_base = '/project_modelview/job_template/api'
    datamodel = SQLAInterface(Project)
    project_type = 'job-template'
    base_filters = [["id", Project_Filter, project_type]]
    label_title = '模板分类'
    edit_form_extra_fields = {
        'type': StringField(
            _(datamodel.obj.lab('type')),
            description="模板分类",
            widget=MyBS3TextFieldWidget(value=project_type, readonly=1),
            default=project_type,
        ),
        'expand': StringField(
            _(datamodel.obj.lab('expand')),
            description='扩展参数。示例参数：<br>"index": 0   表示在pipeline编排中的模板列表的排序位置',
            widget=MyBS3TextAreaFieldWidget(),
            default='{}',
        )
    }
    add_form_extra_fields = edit_form_extra_fields


appbuilder.add_api(Project_ModelView_job_template_Api)


class Project_ModelView_org_Api(Project_ModelView_Base, MyappModelRestApi):
    route_base = '/project_modelview/org/api'
    datamodel = SQLAInterface(Project)
    project_type = 'org'
    base_filters = [["id", Project_Filter, project_type]]
    related_views = [Project_User_ModelView_Api, ]
    label_title = '项目分组'
    edit_form_extra_fields = {
        'type': StringField(
            _(datamodel.obj.lab('type')),
            description="项目分组",
            widget=MyBS3TextFieldWidget(value=project_type, readonly=1),
            default=project_type,
        ),
        'expand': StringField(
            _(datamodel.obj.lab('expand')),
            description='扩展参数。示例参数：<br>"cluster": "dev"<br>"org": "public"<br>"volume_mount": "kubeflow-user-workspace(pvc):/mnt/;/data/k8s/../group1(hostpath):/mnt1"<br>"SERVICE_EXTERNAL_IP":"xx.内网.xx.xx|xx.公网.xx.xx"',
            widget=MyBS3TextAreaFieldWidget(),
            default=json.dumps({"cluster": "dev"}, indent=4, ensure_ascii=False),
        )
    }
    add_form_extra_fields = edit_form_extra_fields

    expand_columns = {
        "expand": {
            "cluster": SelectField(
                label='集群',
                widget=Select2Widget(),
                default='dev',
                description='使用该项目组的所有任务部署到的目的集群',
                choices=[[x, x] for x in list(conf.get('CLUSTERS', {"dev": {}}).keys())],
                validators=[DataRequired()]
            ),
            'volume_mount': StringField(
                label=_('挂载'),
                default='kubeflow-user-workspace(pvc):/mnt/',
                description='使用该项目组的所有任务会自动添加的挂载目录，kubeflow-user-workspace(pvc):/mnt/,/data/k8s/../group1(hostpath):/mnt1,nfs-test(storage):/nfs',
                widget=BS3TextFieldWidget(),
                validators=[]
            ),
            'SERVICE_EXTERNAL_IP': StringField(
                label=_('服务代理ip'),
                default='',
                description='服务的代理ip，xx.内网.xx.xx|xx.公网.xx.xx',
                widget=BS3TextFieldWidget(),
                validators=[]
            )
        }
    }

    def pre_add_web(self):
        self.edit_form_extra_fields['type'] = StringField(
            _(self.datamodel.obj.lab('type')),
            description="项目分组",
            widget=MyBS3TextFieldWidget(value=self.project_type, readonly=1),
            default=self.project_type,
        )
        self.add_form_extra_fields = self.edit_form_extra_fields
        self.expand_columns['expand']['org']=StringField(
            label=_('资源组'),
            widget=BS3TextFieldWidget(),
            default='public',
            description='使用该项目组的所有任务部署到的目的资源组，通过机器label org=xx决定',
            # choices=[['public','public'],['safe','safe']],
            validators=[DataRequired()]
        )

appbuilder.add_api(Project_ModelView_org_Api)


class Project_ModelView_train_model_Api(Project_ModelView_Base, MyappModelRestApi):
    route_base = '/project_modelview/model/api'
    datamodel = SQLAInterface(Project)
    project_type = 'model'
    label_title = '模型分组'
    base_filters = [["id", Project_Filter, project_type]]
    edit_form_extra_fields = {
        'type': StringField(
            _(datamodel.obj.lab('type')),
            description="模型分组",
            widget=MyBS3TextFieldWidget(value=project_type, readonly=1),
            default=project_type,
        )
    }
    add_form_extra_fields = edit_form_extra_fields


appbuilder.add_api(Project_ModelView_train_model_Api)


class Project_ModelView_Api(Project_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Project)
    route_base = '/project_modelview/api'
    related_views = [Project_User_ModelView_Api, ]


appbuilder.add_api(Project_ModelView_Api)
