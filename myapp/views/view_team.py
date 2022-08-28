from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.model_team import Project,Project_User
from flask_appbuilder.actions import action
from wtforms import BooleanField, IntegerField, SelectField, StringField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.models.sqla.filters import FilterEqualFunction, FilterStartsWith,FilterEqual,FilterNotEqual
from wtforms.validators import EqualTo,Length
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.security.decorators import has_access
from myapp.utils import core
from myapp import app, appbuilder,db
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from flask_appbuilder.fieldwidgets import Select2Widget
from myapp.exceptions import MyappException
from myapp import conf, db, get_feature_flags, security_manager,event_logger
from myapp.forms import MyBS3TextFieldWidget,MyBS3TextAreaFieldWidget
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from flask import (
    abort,
    flash,
    g,
    Markup,
    redirect,
    render_template,
    request,
    Response,
    url_for,
)
from .base import (
    api,
    BaseMyappView,
    check_ownership,
    data_payload_response,
    DeleteMixin,
    generate_download_headers,
    get_error_msg,
    get_user_roles,
    handle_api_exception,
    json_error_response,
    json_success,
    MyappFilter,
    MyappModelView,
)
from .baseApi import (
    MyappModelRestApi
)
import pysnooper,datetime,time,json
from flask_appbuilder import CompactCRUDMixin, expose

# table show界面下的
class Project_User_ModelView_Base():
    label_title='组成员'
    datamodel = SQLAInterface(Project_User)
    add_columns = ['project','user','role']
    edit_columns = add_columns
    list_columns = ['user','role']

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
            choices=[[x, x] for x in ['dev', 'ops','creator']],
            description='只有creator可以添加修改组成员，可以添加多个creator',
            validators=[DataRequired()]
        )
    }
    edit_form_extra_fields = add_form_extra_fields

    # @pysnooper.snoop()
    def pre_add(self, item):
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return
        if not g.user.username in item.project.get_creators():
            raise MyappException('just creator can add/edit user')


    def pre_update(self, item):
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return
        if not g.user.username in item.project.get_creators():
            raise MyappException('just creator can add/edit user')

class Project_User_ModelView(Project_User_ModelView_Base,CompactCRUDMixin, MyappModelView):
    datamodel = SQLAInterface(Project_User)

appbuilder.add_view_no_menu(Project_User_ModelView)


class Project_User_ModelView_Api(Project_User_ModelView_Base,MyappModelRestApi):
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
        return query.filter(self.model.type==value).order_by(self.model.id.desc())



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
        return query.filter(Project.type=='org').order_by(Project.id.desc())

    my_user_id = g.user.get_id() if g.user else 0
    owner_ids_query = db.session.query(Project_User.project_id).filter(Project_User.user_id == my_user_id)

    return query.filter(Project.id.in_(owner_ids_query)).filter(Project.type=='org').order_by(Project.id.desc())


class Project_ModelView_Base():
    label_title='项目组'
    datamodel = SQLAInterface(Project)
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    list_columns = ['name','user','type']
    cols_width = {
        "name":{"type": "ellip2", "width": 200},
        "user": {"type": "ellip3", "width": 700},
        "type": {"type": "ellip2", "width": 200},
    }
    related_views = [Project_User_ModelView,]
    add_columns = ['name','describe','expand']
    edit_columns = add_columns


    # @pysnooper.snoop()
    def pre_add_get(self):
        self.edit_form_extra_fields['type'] = StringField(
            _(self.datamodel.obj.lab('type')),
            description="项目分组",
            widget=MyBS3TextFieldWidget(value=self.project_type,readonly=1),
            default=self.project_type,
        )
        self.add_form_extra_fields = self.edit_form_extra_fields


    def pre_update(self, item):
        if not item.type:
            item.type = self.project_type
        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return
        if not g.user.username in item.get_creators():
            raise MyappException('just creator can add/edit')

    # before update, check permission
    def pre_update_get(self, item):
        self.pre_add_get()
        self.check_item_permissions(item)
        if not self.user_permissions['edit']:
            flash('just creator can add/edit user','warning')
            raise MyappException('just creator can add/edit user')

    # check permission
    def check_item_permissions(self,item):
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
        creator = Project_User(role='creator',user=g.user,project=item)
        db.session.add(creator)
        db.session.commit()


    # @pysnooper.snoop()
    def post_list(self,items):
        return core.sort_expand_index(items,db.session)



# class Project_ModelView_job_template(Project_ModelView_Base,MyappModelView):
#     project_type = 'job-template'
#     base_filters = [["id", Project_Filter, project_type]]
#     datamodel = SQLAInterface(Project)
#     label_title = '模板分类'
#


class Project_ModelView_job_template_Api(Project_ModelView_Base,MyappModelRestApi):
    route_base = '/project_modelview/job_template/api'
    datamodel = SQLAInterface(Project)
    project_type = 'job-template'
    base_filters = [["id", Project_Filter, project_type]]
    related_views = [Project_User_ModelView_Api, ]
    label_title = '模板分类'
    edit_form_extra_fields={
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
    add_form_extra_fields=edit_form_extra_fields

appbuilder.add_api(Project_ModelView_job_template_Api)


# class Project_ModelView_org(Project_ModelView_Base,MyappModelView):
#     project_type='org'
#     base_filters = [["id", Project_Filter, project_type]]
#     datamodel = SQLAInterface(Project)
#     label_title = '项目分组'
#
#


class Project_ModelView_org_Api(Project_ModelView_Base,MyappModelRestApi):
    route_base = '/project_modelview/org/api'
    datamodel = SQLAInterface(Project)
    project_type = 'org'
    base_filters = [["id", Project_Filter, project_type]]
    related_views = [Project_User_ModelView_Api, ]
    label_title = '项目分组'
    edit_form_extra_fields={
        'type': StringField(
            _(datamodel.obj.lab('type')),
            description="项目分组",
            widget=MyBS3TextFieldWidget(value=project_type, readonly=1),
            default=project_type,
        ),
        'expand': StringField(
            _(datamodel.obj.lab('expand')),
            description='扩展参数。示例参数：<br>"cluster": "dev"<br>"node_selector": "org=public"<br>"volume_mount": "kubeflow-user-workspace(pvc):/mnt/;/data/k8s/../group1(hostpath):/mnt1;4G(memory):/dev/shm"<br>"SERVICE_EXTERNAL_IP":"xx.xx.xx.xx"',
            widget=MyBS3TextAreaFieldWidget(),
            default=json.dumps({"cluster": "dev"},indent=4,ensure_ascii=False),
        )
    }
    add_form_extra_fields=edit_form_extra_fields


appbuilder.add_api(Project_ModelView_org_Api)



# class Project_ModelView_train_model(Project_ModelView_Base,MyappModelView):
#     project_type = 'model'
#     base_filters = [["id", Project_Filter, project_type]]
#     datamodel = SQLAInterface(Project)
#     label_title = '模型分组'
#
#


class Project_ModelView_train_model_Api(Project_ModelView_Base,MyappModelRestApi):
    route_base = '/project_modelview/model/api'
    datamodel = SQLAInterface(Project)
    project_type = 'model'
    label_title = '模型分组'
    base_filters = [["id", Project_Filter, project_type]]
    related_views = [Project_User_ModelView_Api, ]
    edit_form_extra_fields={
        'type': StringField(
            _(datamodel.obj.lab('type')),
            description="模型分组",
            widget=MyBS3TextFieldWidget(value=project_type, readonly=1),
            default=project_type,
        )
    }
    add_form_extra_fields=edit_form_extra_fields


appbuilder.add_api(Project_ModelView_train_model_Api)



class Project_ModelView_Api(Project_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Project)
    route_base = '/project_modelview/api'
    related_views=[Project_User_ModelView_Api,]

appbuilder.add_api(Project_ModelView_Api)






