from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import re
import urllib.parse
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError

from myapp.models.model_etl_pipeline import ETL_Pipeline,ETL_Task
from myapp.models.model_team import Project,Project_User
from myapp.views.view_team import Project_Join_Filter
from flask_appbuilder.actions import action
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for
from flask_appbuilder.models.sqla.filters import FilterEqualFunction, FilterStartsWith,FilterEqual,FilterNotEqual
from wtforms.validators import EqualTo,Length
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db,event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from jinja2 import Template
from jinja2 import contextfilter
from jinja2 import Environment, BaseLoader, DebugUndefined, StrictUndefined
import os,sys
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from myapp.forms import JsonValidator
from myapp.views.view_task import Task_ModelView
from sqlalchemy import and_, or_, select
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from myapp.project import push_message,push_admin
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget,BS3TextAreaFieldWidget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from myapp.utils.py import py_k8s
from flask_wtf.file import FileField
import shlex
import re,copy
from kubernetes.client.models import (
    V1Container, V1EnvVar, V1EnvFromSource, V1SecurityContext, V1Probe,
    V1ResourceRequirements, V1VolumeDevice, V1VolumeMount, V1ContainerPort,
    V1Lifecycle, V1Volume,V1SecurityContext
)
from .baseApi import (
    MyappModelRestApi
)
from flask import (
    current_app,
    abort,
    flash,
    g,
    Markup,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)
from myapp import security_manager
from myapp.views.view_team import filter_join_org_project

from werkzeug.datastructures import FileStorage
from kubernetes import client as k8s_client
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
    json_response
)

from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config
logging = app.logger
APPGROUP_INFO=['资源组1','资源组2','资源组3']

class ETL_Task_ModelView_Base():
    label_title="任务"
    datamodel = SQLAInterface(ETL_Task)
    check_redirect_list_url = conf.get('MODEL_URLS',{}).get('etl_pipeline')

    base_permissions = ['can_list','can_show','can_delete']
    base_order = ("changed_on", "desc")
    # order_columns = ['id','changed_on']
    order_columns = ['id']
    search_columns = ['name','template','etl_task_id','created_by']
    list_columns = ['template','name','describe','etl_task_id','creator']
    cols_width = {
        "template":{"type": "ellip2", "width": 200},
        "name": {"type": "ellip2", "width": 300},
        "describe": {"type": "ellip2", "width": 300},
        "etl_task_id": {"type": "ellip2", "width": 200},
    }
    spec_lable_columns={
        "template":"功能类型"
    }
    def pre_add_get(self):
        self.default_filter = {
            "created_by": g.user.id
        }
    def post_list(self, items):
        flash('此部分仅提供任务流编排能力，管理员自行对接调度Azkaban/Oozie/Airflow/argo等调度平台能力','warning')
        return items
    show_columns = ['template','name','describe','etl_task_id','created_by','changed_by','created_on','changed_on','task_args']



class ETL_Task_ModelView(ETL_Task_ModelView_Base,MyappModelView):
    datamodel = SQLAInterface(ETL_Task)


appbuilder.add_view_no_menu(ETL_Task_ModelView)

# 添加api
class ETL_Task_ModelView_Api(ETL_Task_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(ETL_Task)
    route_base = '/etl_task_modelview/api'

appbuilder.add_api(ETL_Task_ModelView_Api)



class ETL_Pipeline_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        join_projects_id = security_manager.get_join_projects_id(db.session)
        # public_project_id =
        # logging.info(join_projects_id)
        return query.filter(
            or_(
                self.model.project_id.in_(join_projects_id),
                # self.model.project.name.in_(['public'])
            )
        )



class ETL_Pipeline_ModelView_Base():
    label_title='任务流'
    datamodel = SQLAInterface(ETL_Pipeline)

    base_permissions = ['can_show','can_edit','can_list','can_delete','can_add']
    base_order = ("changed_on", "desc")
    # order_columns = ['id','changed_on']
    order_columns = ['id']

    list_columns = ['project','etl_pipeline_url','creator','modified']
    cols_width = {
        "project":{"type": "ellip2", "width": 200},
        "etl_pipeline_url": {"type": "ellip2", "width": 400},
        "creator": {"type": "ellip2", "width": 100},
        "modified": {"type": "ellip2", "width": 100},
    }

    add_columns = ['project','name','describe']
    show_columns = ['project','name','describe','config_html','dag_json_html','created_by','changed_by','created_on','changed_on','expand_html']
    edit_columns = add_columns


    base_filters = [["id", ETL_Pipeline_Filter, lambda: []]]
    conv = GeneralModelConverter(datamodel)

    # related_views = [ETL_Task_ModelView,]

    add_form_extra_fields = {
        "name": StringField(
            _(datamodel.obj.lab('name')),
            description="英文名(小写字母、数字、- 组成)，最长50个字符",
            default='',
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54),DataRequired()]
        ),
        "project":QuerySelectField(
            _(datamodel.obj.lab('project')),
            query_factory=filter_join_org_project,
            allow_blank=True,
            widget=Select2Widget()
        ),
        "describe": StringField(
            _(datamodel.obj.lab('describe')),
            default='',
            widget=BS3TextFieldWidget(),
            description="任务流描述",
        ),
        "dag_json": StringField(
            _(datamodel.obj.lab('dag_json')),
            default='{}',
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        )
    }


    edit_form_extra_fields = add_form_extra_fields


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

    # @pysnooper.snoop()
    def pre_add(self, item):
        if not item.dag_json:
            item.dag_json='{}'
        item.name = item.name.replace('_', '-')[0:54].lower().strip('-')

    # @pysnooper.snoop()
    def pre_update(self, item):

        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        else:
            item.expand='{}'

        if item.dag_json:
            dag_json = json.loads(item.dag_json)
            for task_name in copy.deepcopy(dag_json):
                if 'templte_common_ui_config' in dag_json[task_name]:
                    del dag_json[task_name]['templte_common_ui_config']
                    del dag_json[task_name]['templte_ui_config']
                # if 'templte_common_ui_config' in dag_json[task_name]:

            for node_name in dag_json:
                if not dag_json[node_name].get('task_id',''):
                    dag_json[node_name]['task_id']=uuid.uuid4().hex[:6]
            item.dag_json = json.dumps(dag_json,indent=4,ensure_ascii=False)
        item.name = item.name.replace('_', '-')[0:54].lower()


    # 删除前先把下面的task删除了
    # @pysnooper.snoop()
    def pre_delete(self, pipeline):
        flash('此处仅删除本地元数据，请及时删除远程任务','warning')
        # 删除本地
        exist_tasks = db.session.query(ETL_Task).filter_by(etl_pipeline_id=pipeline.id).all()
        for exist_task in exist_tasks:
            db.session.delete(exist_task)
            db.session.commit()


    # @pysnooper.snoop(watch_explode=('dag_json',))
    def fix_pipeline_task(self,etl_pipeline):
        if not etl_pipeline:
            return
        dag_json = json.loads(etl_pipeline.dag_json) if etl_pipeline.dag_json else {}
        # print(dag_json)
        task_ids = [int(dag_json[task_name].get('task_id','')) for task_name in dag_json if dag_json[task_name].get('task_id','')]

        exist_tasks = db.session.query(ETL_Task).filter_by(etl_pipeline_id=etl_pipeline.id).all()

        # 删除已经删除的task
        for exist_task in exist_tasks:
            # print(exist_task.id,task_ids)
            if exist_task.id not in task_ids:
                db.session.delete(exist_task)
                db.session.commit()

        # 添加新的task和更新旧的
        for task_name in dag_json:
            task_id = dag_json[task_name].get('task_id','')
            task_args = dag_json[task_name].get('task-config', {})
            if task_id:
                exist_task = db.session.query(ETL_Task).filter_by(etl_pipeline_id=etl_pipeline.id).filter_by(id=int(task_id)).first()
                if exist_task:
                    exist_task.name = task_name
                    exist_task.describe=dag_json[task_name].get('label','')
                    exist_task.template = dag_json[task_name].get('template', '')
                    exist_task.task_args = json.dumps(task_args)
                    exist_task.etl_task_id = dag_json[task_name].get('etl_task_id', '')
                    db.session.commit()
            else:
                etl_task = ETL_Task(
                    name=task_name,
                    describe=dag_json[task_name].get('label', ''),
                    template=dag_json[task_name].get('template', ''),
                    task_args = json.dumps(task_args),
                    etl_task_id=dag_json[task_name].get('etl_task_id', ''),
                    etl_pipeline_id=etl_pipeline.id
                )
                db.session.add(etl_task)
                db.session.commit()
                dag_json[task_name]['task_id']=etl_task.id

        etl_pipeline.dag_json = json.dumps(dag_json,indent=4,ensure_ascii=False)
        db.session.commit()

        pass
        pass

    @expose("/config/<etl_pipeline_id>",methods=("GET",'POST'))
    # @pysnooper.snoop()
    def pipeline_config(self,etl_pipeline_id):
        print(etl_pipeline_id)
        pipeline = db.session.query(ETL_Pipeline).filter_by(id=etl_pipeline_id).first()

        if not pipeline:
            return jsonify({
                "status":1,
                "message":"任务流不存在",
                "result":{}
            })
        if request.method.lower()=='post':

            if g.user.username != pipeline.created_by.username and not g.user.is_admin():
                return jsonify({
                    "result": {},
                    "message": "只有创建者或管理员可修改",
                    "status": -1
                })

            req_data = request.get_json()
            if 'config' in req_data:
                pipeline.config = json.dumps(req_data['config'], indent=4, ensure_ascii=False)
            if 'dag_json' in req_data and type(req_data['dag_json'])==dict:

                new_dag_json = json.loads(pipeline.dag_json) if pipeline.dag_json else {}
                # 把新节点加进去，因为有时候前端不保留部分字段，只保留在后端
                for task_name in req_data['dag_json']:
                    if task_name not in new_dag_json:
                        new_dag_json[task_name]=req_data['dag_json'][task_name]

                # 把旧节点更新，因为有时候前端不保留部分字段，只保留在后端，但是注意前端可能会删除自己管理的节点，这部分参数要保留
                for task_name in copy.deepcopy(new_dag_json):
                    if task_name not in req_data['dag_json']:
                        del new_dag_json[task_name]
                    else:
                        task_config= req_data['dag_json'][task_name].get('task-config',{})
                        if task_config and 'crontab' in task_config:
                            new_dag_json[task_name].update(req_data['dag_json'][task_name])

                # 校验一下，把不存在的上游节点更新掉
                for task_name in new_dag_json:
                    task = new_dag_json[task_name]
                    upstreams_nodes_name = task.get('upstream', [])
                    new_upstreams_nodes_name=[]
                    for name in upstreams_nodes_name:
                        if name in new_dag_json:
                            new_upstreams_nodes_name.append(name)
                    new_dag_json[task_name]['upstream']=new_upstreams_nodes_name


                pipeline.dag_json = json.dumps(new_dag_json, indent=4, ensure_ascii=False)

            db.session.commit()
            self.fix_pipeline_task(pipeline)

        back_dag_json = json.loads(pipeline.dag_json)
        # 更新任务 操作 按钮
        for task_name in back_dag_json:
            task = back_dag_json[task_name]
            back_dag_json[task_name]["task_jump_button"] = []
            etl_task_id=task.get('etl_task_id','')
            if etl_task_id:
                back_dag_json[task_name]["task_jump_button"].append(
                    {
                        "name": "任务查看",
                        "action_url": conf.get('MODEL_URLS', {}).get('etl_task') + '?taskId=' + etl_task_id,
                        "icon_svg": '<svg t="1660558833880" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2441" width="200" height="200"><path d="M831.825474 63.940169H191.939717C121.2479 63.940169 63.940169 121.2479 63.940169 191.939717v639.885757C63.940169 902.517291 121.2479 959.825022 191.939717 959.825022h639.885757c70.691817 0 127.999548-57.307731 127.999548-127.999548V191.939717C959.825022 121.2479 902.517291 63.940169 831.825474 63.940169zM895.884854 831.998871A63.835408 63.835408 0 0 1 831.912173 895.884854H192.087827c-17.112123 0-33.270563-6.574639-45.372232-18.67631S127.880338 849.110994 127.880338 831.998871V192.001129A64.236389 64.236389 0 0 1 192.087827 127.880338h639.824346A64.037705 64.037705 0 0 1 895.884854 192.001129v639.997742z" fill="#225ed2" p-id="2442"></path><path d="M791.998335 351.851551h-255.999097a31.970084 31.970084 0 0 0 0 63.940169h255.999097a31.970084 31.970084 0 0 0 0-63.940169zM791.998335 607.973471h-255.999097a31.970084 31.970084 0 0 0 0 63.940169h255.999097a31.970084 31.970084 0 0 0 0-63.940169zM344.001722 527.997686c-61.855792 0-111.985607 50.144265-111.985607 111.985606s50.144265 111.985607 111.985607 111.985607 111.985607-50.144265 111.985606-111.985607-50.129815-111.985607-111.985606-111.985606z m33.982213 145.982269a48.045438 48.045438 0 1 1 14.088511-33.982213 47.745605 47.745605 0 0 1-14.088511 33.985826zM417.395643 297.394035L311.999125 402.78694 270.6078 361.392003a31.970084 31.970084 0 1 0-45.213286 45.213285l63.997968 64.001581a31.970084 31.970084 0 0 0 45.213286 0l127.999548-127.999549a31.970084 31.970084 0 0 0-45.209673-45.213285z" fill="#225ed2" p-id="2443"></path></svg>'
                    }
                )
                back_dag_json[task_name]["task_jump_button"].append(
                    {
                        "name": "任务实例",
                        "action_url": conf.get('MODEL_URLS', {}).get('etl_task_instance') + "?taskId=" + etl_task_id,
                        "icon_svg": '<svg t="1660554835088" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2435" width="200" height="200"><path d="M112.64 95.36a32 32 0 0 0-32 32v332.16a32 32 0 0 0 32 32h332.16a32 32 0 0 0 32-32V128a32 32 0 0 0-32-32z m300.16 332.16H144.64V159.36h268.16zM938.88 293.76a197.76 197.76 0 1 0-197.76 197.76 198.4 198.4 0 0 0 197.76-197.76z m-332.16 0a133.76 133.76 0 1 1 133.76 133.76 134.4 134.4 0 0 1-133.76-133.76zM99.84 928.64h365.44a32 32 0 0 0 27.52-48L310.4 563.84a33.28 33.28 0 0 0-55.68 0l-182.4 316.8a32 32 0 0 0 27.52 48z m182.4-284.16l128 220.16h-256zM832 552.96h-177.28a32 32 0 0 0-27.52 16l-89.6 155.52a32 32 0 0 0 0 32l89.6 155.52a32 32 0 0 0 27.52 16H832a32 32 0 0 0 27.52-16l89.6-155.52a32 32 0 0 0 0-32l-89.6-155.52a32 32 0 0 0-27.52-16z m-18.56 311.04h-140.16L601.6 741.12l71.68-123.52h142.72l71.68 123.52z" fill="#225ed2" p-id="2436"></path></svg>'
                    }
                )

        config = {
            "id":pipeline.id,
            "name":pipeline.name,
            "label":pipeline.describe,
            "project":pipeline.project.describe,
            "pipeline_ui_config":{
                "alert":{
                    "alert_user":{
                        "type": "str",
                        "item_type": "str",
                        "label": "报警用户",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "报警用户名，逗号分隔",
                        "describe": "报警用户，逗号分隔",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            },
            "pipeline_jump_button": [
                {
                    "name": "调度实例",
                    "action_url": conf.get('MODEL_URLS', {}).get('etl_task'),
                    "icon_svg": '<svg t="1660554835088" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2435" width="200" height="200"><path d="M112.64 95.36a32 32 0 0 0-32 32v332.16a32 32 0 0 0 32 32h332.16a32 32 0 0 0 32-32V128a32 32 0 0 0-32-32z m300.16 332.16H144.64V159.36h268.16zM938.88 293.76a197.76 197.76 0 1 0-197.76 197.76 198.4 198.4 0 0 0 197.76-197.76z m-332.16 0a133.76 133.76 0 1 1 133.76 133.76 134.4 134.4 0 0 1-133.76-133.76zM99.84 928.64h365.44a32 32 0 0 0 27.52-48L310.4 563.84a33.28 33.28 0 0 0-55.68 0l-182.4 316.8a32 32 0 0 0 27.52 48z m182.4-284.16l128 220.16h-256zM832 552.96h-177.28a32 32 0 0 0-27.52 16l-89.6 155.52a32 32 0 0 0 0 32l89.6 155.52a32 32 0 0 0 27.52 16H832a32 32 0 0 0 27.52-16l89.6-155.52a32 32 0 0 0 0-32l-89.6-155.52a32 32 0 0 0-27.52-16z m-18.56 311.04h-140.16L601.6 741.12l71.68-123.52h142.72l71.68 123.52z" fill="#225ed2" p-id="2436"></path></svg>'
                },
                {
                    "name": "监控",
                    "action_url": conf.get('MODEL_URLS', {}).get('etl_task'),
                    "icon_svg": '<svg t="1660554975153" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3500" width="200" height="200"><path d="M512 302.037c-83.338 0-161.688 32.454-220.617 91.383C232.454 452.348 200 530.698 200 614.036c0 17.673 14.327 32 32 32s32-14.327 32-32c0-66.243 25.796-128.521 72.638-175.362 46.841-46.841 109.119-72.638 175.362-72.638s128.521 25.796 175.362 72.638C734.203 485.515 760 547.793 760 614.036c0 17.673 14.327 32 32 32s32-14.327 32-32c0-83.338-32.454-161.688-91.383-220.617C673.688 334.49 595.338 302.037 512 302.037z" fill="#225ed2" p-id="3501"></path><path d="M512 158C264.576 158 64 358.576 64 606c0 89.999 26.545 173.796 72.224 244h751.553C933.455 779.796 960 695.999 960 606c0-247.424-200.576-448-448-448z m339.288 628H172.712C143.373 730.813 128 669.228 128 606c0-51.868 10.144-102.15 30.15-149.451 19.337-45.719 47.034-86.792 82.321-122.078 35.286-35.287 76.359-62.983 122.078-82.321C409.85 232.144 460.132 222 512 222c51.868 0 102.15 10.144 149.451 30.15 45.719 19.337 86.792 47.034 122.078 82.321 35.287 35.286 62.983 76.359 82.321 122.078C885.856 503.85 896 554.132 896 606c0 63.228-15.373 124.813-44.712 180z" fill="#225ed2" p-id="3502"></path><path d="M532.241 586.466a79.753 79.753 0 0 0-29.217 5.529l-69.087-69.087c-12.497-12.499-32.758-12.497-45.255-0.001-12.497 12.497-12.497 32.759 0 45.255l69.088 69.088a79.753 79.753 0 0 0-5.529 29.217c0 44.183 35.817 80 80 80s80-35.817 80-80-35.818-80.001-80-80.001z" fill="#225ed2" p-id="3503"></path></svg>'
                }
            ],
            "pipeline_run_button": [
                {
                    "name": "提交",
                    "action_url": "/etl_pipeline_modelview/run_etl_pipeline/%s"%pipeline.id,
                    "icon_svg": '<svg t="1660558913467" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5032" width="200" height="200"><path d="M689.3568 820.9408 333.6192 820.9408c-13.9264 0-25.1904-11.264-25.1904-25.1904L308.4288 465.152l-211.968 0c-10.1888 0-19.4048-6.144-23.296-15.5648C69.2224 440.2176 71.424 429.312 78.6432 422.144l415.0272-415.0784c9.472-9.472 26.1632-9.472 35.6352 0l411.392 411.3408c7.2704 4.4544 12.0832 12.4416 12.0832 21.5552 0 14.1312-11.52 24.576-25.7024 25.1904-0.1536 0-0.3072 0-0.512 0l-211.968 0 0 330.5472C714.5984 809.6256 703.2832 820.9408 689.3568 820.9408zM358.8096 770.5088l305.3568 0L664.1664 439.9616c0-13.9264 11.264-25.1904 25.1904-25.1904l176.3328 0L511.488 60.5184 157.2864 414.7712l176.3328 0c13.9264 0 25.1904 11.264 25.1904 25.1904L358.8096 770.5088z" p-id="5033" fill="#225ed2"></path><path d="M96.4096 923.1872l830.1056 0L926.5152 1024 96.4096 1024 96.4096 923.1872 96.4096 923.1872zM96.4096 923.1872" p-id="5034" fill="#225ed2"></path></svg>'
                },
                {
                    "name": "任务查看",
                    "action_url": conf.get('MODEL_URLS', {}).get('etl_task'),
                    "icon_svg": '<svg t="1660558833880" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2441" width="200" height="200"><path d="M831.825474 63.940169H191.939717C121.2479 63.940169 63.940169 121.2479 63.940169 191.939717v639.885757C63.940169 902.517291 121.2479 959.825022 191.939717 959.825022h639.885757c70.691817 0 127.999548-57.307731 127.999548-127.999548V191.939717C959.825022 121.2479 902.517291 63.940169 831.825474 63.940169zM895.884854 831.998871A63.835408 63.835408 0 0 1 831.912173 895.884854H192.087827c-17.112123 0-33.270563-6.574639-45.372232-18.67631S127.880338 849.110994 127.880338 831.998871V192.001129A64.236389 64.236389 0 0 1 192.087827 127.880338h639.824346A64.037705 64.037705 0 0 1 895.884854 192.001129v639.997742z" fill="#225ed2" p-id="2442"></path><path d="M791.998335 351.851551h-255.999097a31.970084 31.970084 0 0 0 0 63.940169h255.999097a31.970084 31.970084 0 0 0 0-63.940169zM791.998335 607.973471h-255.999097a31.970084 31.970084 0 0 0 0 63.940169h255.999097a31.970084 31.970084 0 0 0 0-63.940169zM344.001722 527.997686c-61.855792 0-111.985607 50.144265-111.985607 111.985606s50.144265 111.985607 111.985607 111.985607 111.985607-50.144265 111.985606-111.985607-50.129815-111.985607-111.985606-111.985606z m33.982213 145.982269a48.045438 48.045438 0 1 1 14.088511-33.982213 47.745605 47.745605 0 0 1-14.088511 33.985826zM417.395643 297.394035L311.999125 402.78694 270.6078 361.392003a31.970084 31.970084 0 1 0-45.213286 45.213285l63.997968 64.001581a31.970084 31.970084 0 0 0 45.213286 0l127.999548-127.999549a31.970084 31.970084 0 0 0-45.209673-45.213285z" fill="#225ed2" p-id="2443"></path></svg>'
                }
            ],
            "dag_json":back_dag_json,
            "config": json.loads(pipeline.config),
            "message": "success",
            "status": 0
        }

        return jsonify(config)

    all_template = {
        "message": "success",
        "task_metadata_ui_config": {
            "metadata": {
                "label": {
                    "type": "str",
                    "item_type": "str",
                    "label": "中文名称",
                    "require": 1,
                    "choice": [],
                    "range": "",
                    "default": "",
                    "placeholder": "",
                    "describe": "任务中文别名",
                    "editable": 1,
                    "addable": 1,
                    "condition": "",
                    "sub_args": {}
                }
            }
        },
        "templte_common_ui_config": {
            "任务元数据": {
                "crontab": {
                    "type": "str",
                    "item_type": "str",
                    "label": "调度周期",
                    "require": 0,
                    "choice": [],
                    "range": "",
                    "default": "1 1 * * *",
                    "placeholder": "",
                    "describe": "周期任务的时间设定 * * * * * 一次性任务可不填写 <br>表示为 minute hour day month week",
                    "editable": 1,
                    "addable": 0,  # 1 为仅在添加时可修改
                    "condition": "",
                    "sub_args": {}
                },
                "selfDepend": {
                    "type": "str",
                    "item_type": "str",
                    "label": "自依赖判断",
                    "require": 1,
                    "choice": ["自依赖", '单实例运行', '多实例运行'],
                    "range": "",
                    "default": "单实例运行",
                    "placeholder": "",
                    "describe": "一个任务的多次调度实例之间是否要进行前后依赖",
                    "editable": 1,
                    "addable": 0, # 1 为仅在添加时可修改
                    "condition": "",
                    "sub_args": {}
                },
                "hiveAppGroup": {
                    "type": "str",
                    "item_type": "str",
                    "label": "资源组",
                    "require": 1,
                    "choice": [item for item in APPGROUP_INFO],
                    "range": "",
                    "default": [item for item in APPGROUP_INFO][0],
                    "placeholder": "",
                    "describe": "资源组",
                    "editable": 1,
                    "addable": 0,
                    "condition": "",
                    "sub_args": {}
                }
            },
            "监控配置": {
                "alert_user": {
                    "type": "str",
                    "item_type": "str",
                    "label": "报警用户",
                    "require": 0,
                    "choice": [],
                    "range": "",
                    "default": "admin,",
                    "placeholder": "",
                    "describe": "报警用户，逗号分隔",
                    "editable": 1,
                    "addable": 0,  # 1 为仅在添加时可修改
                    "condition": "",
                    "sub_args": {}
                },
                "timeout": {
                    "type": "str",
                    "item_type": "str",
                    "label": "超时中断",
                    "require": 1,
                    "choice": [],
                    "range": "",
                    "default": "0",
                    "placeholder": "",
                    "describe": "task运行时长限制，为0表示不限制(单位s)",
                    "editable": 1,
                    "addable": 0,  # 1 为仅在添加时可修改
                    "condition": "",
                    "sub_args": {}
                },
                "retry": {
                    "type": "str",
                    "item_type": "str",
                    "label": "重试次数",
                    "require": 1,
                    "choice": [],
                    "range": "",
                    "default": '0',
                    "placeholder": "",
                    "describe": "重试次数",
                    "editable": 1,
                    "addable": 0,
                    "condition": "",
                    "sub_args": {}
                }
            },

        },
        "template_group_order": ["绑定任务", "出库入库", "数据计算", "其他"],
        "templte_list": {
            "绑定任务": [
                {
                    "template_name": "已存在任务",
                    "templte_ui_config": {
                        "参数": {
                            "etl_task_id": {
                                "type": "str",
                                "item_type": "str",
                                "label": "已存在任务的us task id",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "已存在任务的us task id",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "绑定任务",
                    "describe": "绑定已存在任务，类似于创建软链接",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": "xx平台任务流",
                    "templte_ui_config": {
                        "参数": {
                            "pipeline_id": {
                                "type": "str",
                                "item_type": "str",
                                "label": "xx平台任务流id",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "xx平台任务流id，可以在任务流详情处查看。",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "xx平台任务流",
                    "describe": "绑定xx平台任务流，类似于创建软链接。用于创建依赖。",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                }
            ],
            "出库入库": [
                {
                    "template_name": "hdfs入库至hive",
                    "templte_ui_config": {
                        "参数": {
                            "charSet": {
                                "type": "str",
                                "item_type": "str",
                                "label": "源文件字符集",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "UTF-8",
                                "placeholder": "",
                                "describe": "源文件字符集",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "databaseName": {
                                "type": "str",
                                "item_type": "str",
                                "label": "hive数据库名称",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "hive数据库名称",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "tableName": {
                                "type": "str",
                                "item_type": "str",
                                "label": "hive表名",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "hive表名",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "delimiter": {
                                "type": "str",
                                "item_type": "str",
                                "label": "源文件分隔符, 填ascii码",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "9",
                                "placeholder": "",
                                "describe": "默认TAB，ascii码：9",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "failedOnZeroWrited": {
                                "type": "str",
                                "item_type": "str",
                                "label": "入库为空时任务处理",
                                "require": 1,
                                "choice": ["1", "0"],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": "无源文件或入库记录为0时,可以指定任务为成功(0)或失败(1)，默认失败(1)",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "partitionType": {
                                "type": "str",
                                "item_type": "str",
                                "label": "分区格式",
                                "require": 1,
                                "choice": ["P_${YYYYMM}", "P_${YYYYMMDD}", "P_${YYYYMMDDHH}", "NULL"],
                                "range": "",
                                "default": "P_${YYYYMMDDHH}",
                                "placeholder": "",
                                "describe": "分区格式：P_${YYYYMM}、P_${YYYYMMDD}、P_${YYYYMMDDHH}、NULL",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "sourceFilePath": {
                                "type": "str",
                                "item_type": "str",
                                "label": "数据文件hdfs路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "支持三种日期变量:${YYYYMM}、${YYYYMMDD}、${YYYYMMDDHH}。系统用任务实例的数据时间替换日期变量。",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "sourceFileNames": {
                                "type": "str",
                                "item_type": "str",
                                "label": "源文件名",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "*",
                                "placeholder": "",
                                "describe": "源文件名(支持通配符*和${YYYYMMDD});入库不做检查",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "sourceColumnNames": {
                                "type": "str",
                                "item_type": "str",
                                "label": "源文件的栏位名称",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "源文件的栏位名称，以逗号分割（结尾不能是逗号）,必须保证列数和文件内容一致（创建临时表所用表列名）。例如column1,column2,column3。注：不允许输入空格，源文件栏位名称只由大小写字符、数字和下划线组成",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "targetColumnNames": {
                                "type": "str",
                                "item_type": "str",
                                "label": "字段映射关系，即hive表的列名",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "字段映射关系，即hive表的列名。",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "loadMode": {
                                "type": "str",
                                "item_type": "str",
                                "label": "数据入库模式",
                                "require": 1,
                                "choice": ["TRUNCATE", "APPEND"],
                                "range": "",
                                "default": "TRUNCATE",
                                "placeholder": "",
                                "describe": "数据入库模式,TRUNCATE或APPEND;",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "hdfs入库至hive任务",
                    "describe": "hdfs入库至hive任务",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": "hive出库至hdfs",
                    "templte_ui_config": {
                        "参数": {
                            "databaseName": {
                                "type": "str",
                                "item_type": "str",
                                "label": "hive表所在的database",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "hive表所在的database",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "destCheckFileName": {
                                "type": "str",
                                "item_type": "str",
                                "label": "对账文件名",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "对账文件名",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "destCheckFilePath": {
                                "type": "str",
                                "item_type": "str",
                                "label": "对账文件路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "对账文件路径",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "destFileDelimiter": {
                                "type": "str",
                                "item_type": "str",
                                "label": "出库文件分隔符",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "9",
                                "placeholder": "",
                                "describe": "出库文件分隔符，填ascii字符对应的数字。默认TAB：9",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "destFilePath": {
                                "type": "str",
                                "item_type": "str",
                                "label": "出库文件路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "出库文件路径",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "filterSQL": {
                                "type": "text",
                                "item_type": "sql",
                                "label": "源SQL",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 'select t1,t2,t3 from your_table where imp_date=${YYYYMMDD}',
                                "placeholder": "",
                                "describe": "源SQL",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "hive出库至hdfs任务",
                    "describe": "hive出库至hdfs任务",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": "hdfs导入cos",
                    "templte_ui_config": {
                        "参数": {
                            "hdfsPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": "hdfs文件路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "hdfs://xx/xxx",
                                "placeholder": "",
                                "describe": "源hdfs文件路径，包括文件名，支持通配符*，支持${YYYYMMDD}等的日期变量。如果没hdfs路径权限，联系平台管理员。",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "cosPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": "目标cos文件路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "/xx/xx/${YYYYMMDD}.tar.gz",
                                "placeholder": "",
                                "describe": "目标cos文件路径，需包括文件名，支持${YYYYMMDD}等的日期变量，如果有多个文件上传，会在自动在cos文件名后面添加一个随机串。",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "ifNeedZip": {
                                "type": "str",
                                "item_type": "str",
                                "label": "是否需要压缩",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": "是否需要压缩 {0:不需要,1:需要}。压缩会压缩成单个文件。压缩方式为.tar.gz",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "hdfs导入cos/oss/obs",
                    "describe": "hdfs导入cos/oss/obs，基于us的调用shell脚本任务类型实现",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": "cos导入hdfs",
                    "templte_ui_config": {
                        "参数": {
                            "hdfsPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": "hdfs文件路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "hdfs://xx/xxx",
                                "placeholder": "",
                                "describe": "目标hdfs文件路径，不包括文件名，支持${YYYYMMDD}等的日期变量。如果没hdfs路径权限，先联系平台管理员。",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "cosPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": "源cos文件路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "/xx/${YYYYMMDD}.tar.gz",
                                "placeholder": "",
                                "describe": "源cos文件路径，需包括文件名，支持${YYYYMMDD}等的日期变量。如果有多个文件上传，先打成一个.tar.gz压缩包。",
                                "editable": 1,
                                "condition": "", 
                                "sub_args": {}
                            },
                            "ifNeedZip": {
                                "type": "str",
                                "item_type": "str",
                                "label": "是否需要解压缩",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": "是否需要解压缩 {0:不需要,1:需要}。解压方式为tar zcvf。解压后文件会放在目标文件夹。",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "cos/oss/obs导入hdfs",
                    "describe": "cos/oss/obs导入hdfs，基于us的调用shell脚本任务类型实现",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
            ],
            "数据计算": [
                {
                    "template_name": "SparkScala",
                    "templte_ui_config": {
                        "参数": {
                            "jar_path": {
                                "type": "text",
                                "item_type": "str",
                                "label": "jar包路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "填写jar包在notebook里的路径，示例/mnt/admin/pipeline_test.py",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "className": {
                                "type": "str",
                                "item_type": "str",
                                "label": "jar包中主类的名字",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "jar包中主类的名字",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "files": {
                                "type": "text",
                                "item_type": "str",
                                "label": "资源文件列表(--files or --archieves)",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "暂未支持",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "programSpecificParams": {
                                "type": "text",
                                "item_type": "str",
                                "label": "传递给程序的参数",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "传递给程序的参数,空格分隔,不要换行",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "options": {
                                "type": "text",
                                "item_type": "str",
                                "label": "spark扩展参数",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": '''
    选项（spark-submit的--conf参数)。不带分号，使用换行分隔(例如):
    spark.driver.maxResultSize=15G
    spark.driver.cores=4
    spark支持一系列--conf扩展属性，此处可以直接填写。例如：spark.yarn.am.waitTime=100s。
    提交任务时后台会将参数带上提交。换行分隔！！
    ''',
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "dynamicAllocation": {
                                "type": "choice",
                                "item_type": "str",
                                "label": "是否动态资源分配",
                                "require": 1,
                                "choice": ["1", "0"],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": "是否动态资源分配，是：1；否：0",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "driver_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": "driver内存大小",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": "driver内存大小",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "num_executors": {
                                "type": "str",
                                "item_type": "str",
                                "label": "executor数量",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "4",
                                "placeholder": "",
                                "describe": "executor数量",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "executor_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": "executor内存大小",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": "executor内存大小",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },

                            "executor_cores": {
                                "type": "str",
                                "item_type": "str",
                                "label": "executor核心数",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2",
                                "placeholder": "",
                                "describe": "executor核心数",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },

                            "task.main.timeout": {
                                "type": "str",
                                "item_type": "str",
                                "label": "超时时间，单位分钟",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "480",
                                "placeholder": "",
                                "describe": "超时时间，单位分钟：480 (代表8小时)",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "task.check.timeout": {
                                "type": "str",
                                "item_type": "str",
                                "label": "check超时时间，单位分钟",
                                "require": 1,
                                "choice": ["5", "10"],
                                "range": "",
                                "default": "5",
                                "placeholder": "",
                                "describe": "check超时时间，单位分钟",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "SparkScala",
                    "describe": "SparkScala计算",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": "SQL",
                    "templte_ui_config": {
                        "参数": {
                            "filterSQL": {
                                "type": "text",
                                "item_type": "sql",
                                "label": "计算加工逻辑",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": '''
    --库名，替换下面的demo_database
    use demo_database;

    --建表语句，替换下面的demo_table，修改字段。一定要加“if not exists”，这样使只在第一次运行时建表
    CREATE TABLE if not exists demo_table(
        qimei36 STRING COMMENT '唯一设备ID',
        userid_id STRING COMMENT '用户id（各app的用户id）',
        device_id STRING COMMENT '设备id（各app的device_id）',
        platform INT COMMENT '平台（1.ANDROID、2.IOS、3.PC）',
        fav_target_id STRING COMMENT '目标歌单ID',
        songid BIGINT COMMENT '歌曲ID，中央曲库tmeid',
        fav_type INT COMMENT '收藏类型：1.收藏 2.取消收藏',
        ftime INT COMMENT '数据分区时间 格式：yyyymmdd'
    )
    PARTITION BY LIST( ftime )          --定义分区字段，替换掉ftime。
    (
        PARTITION p_20220323 VALUES IN ( 20220323 ),       --初始分区，分区名替换p_20220323，分区值替换20220323
        PARTITION default
    )
    STORED AS ORCFILE COMPRESS;

    -- 分区，根据时间参数新建分区。
    alter table demo_table drop partition (p_${YYYYMMDD});
    alter table demo_table add partition p_${YYYYMMDD} values in (${YYYYMMDD});

    -- 写入，用你的sql逻辑替换。
    insert table demo_table
    select * from other_db::other_table partition(p_${YYYYMMDD}) t;
    ''',
                                "placeholder": "",
                                "describe": "从hive导出数据的sql，比如 select a,b,c FROM table where imp_date='${YYYYMMDD}' ;sql末尾不要用分号结尾",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },

                            "special_para": {
                                "type": "str",
                                "item_type": "str",
                                "label": "hive特殊参数",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "set hive.exec.parallel = true;set hive.execute.engine=spark;set hive.multi.join.use.hive=false;set hive.spark.failed.retry=false;",
                                "placeholder": "",
                                "describe": "hive特殊参数",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "sql执行",
                    "describe": "sql执行",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": "pyspark",
                    "templte_ui_config": {
                        "参数": {
                            "py_script_path": {
                                "type": "str",
                                "item_type": "str",
                                "label": "pyspark脚本路径",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "填写pyspark脚本在notebook里的路径，示例/mnt/admin/pipeline_test.py",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "files": {
                                "type": "str",
                                "item_type": "str",
                                "label": "资源文件列表",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "暂未支持",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "pyFiles": {
                                "type": "str",
                                "item_type": "str",
                                "label": "执行脚本依赖文件列表(spark-submit中的--py-files)",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "暂未支持",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "programSpecificParams": {
                                "type": "text",
                                "item_type": "str",
                                "label": "传递给程序的参数",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "传递给程序的参数,空格分隔,不要换行",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "options": {
                                "type": "text",
                                "item_type": "str",
                                "label": "spark扩展参数",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": '',
                                "placeholder": "",
                                "describe": "选项（spark-submit的--conf参数)。例如：spark.yarn.am.waitTime=100s。换行分隔！！",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "dynamicAllocation": {
                                "type": "str",
                                "item_type": "str",
                                "label": "是否动态资源分配",
                                "require": 1,
                                "choice": [1, 0],
                                "range": "",
                                "default": 1,
                                "placeholder": "",
                                "describe": "是否动态资源分配，是：1；否：0",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "driver_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": "driver内存大小",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": "driver内存大小",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "num_executors": {
                                "type": "str",
                                "item_type": "str",
                                "label": "executor数量",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 4,
                                "placeholder": "",
                                "describe": "executor数量",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },

                            "executor_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": "executor内存大小",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": "executor内存大小",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },

                            "executor_cores": {
                                "type": "str",
                                "item_type": "str",
                                "label": "executor核心数",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 2,
                                "placeholder": "",
                                "describe": "executor核心数",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },

                            "task.main.timeout": {
                                "type": "str",
                                "item_type": "str",
                                "label": "超时时间，单位分钟",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 480,
                                "placeholder": "",
                                "describe": "超时时间，单位分钟：480 (代表8小时)",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "task.check.timeout": {
                                "type": "int",
                                "item_type": "int",
                                "label": "check超时时间，单位分钟",
                                "require": 1,
                                "choice": ["5", "10"],
                                "range": "",
                                "default": "5",
                                "placeholder": "",
                                "describe": "check超时时间，单位分钟",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            }
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "pyspark",
                    "describe": "pyspark脚本执行",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                }
            ],
            "其他": [
                {
                    "template_name": "test",
                    "templte_ui_config": {
                        "参数": {
                            "args1": {
                                "type": "json",
                                "item_type": "str",
                                "label": "参数1",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "参数1",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "args2": {
                                "type": "str",
                                "item_type": "str",
                                "label": "参数2",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "参数2",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "args3": {
                                "type": "int",
                                "item_type": "str",
                                "label": "参数3",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "参数3",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "args4": {
                                "type": "choice",
                                "item_type": "str",
                                "label": "参数4",
                                "require": 1,
                                "choice": ["aa", "bb", "cc", "dd"],
                                "range": "",
                                "default": "aa",
                                "placeholder": "",
                                "describe": "参数4",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "args5": {
                                "type": "str",
                                "item_type": "str",
                                "label": "参数5",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "这是个不可编辑参数",
                                "placeholder": "",
                                "describe": "这个参数不可编辑",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                            "args6": {
                                "type": "text",
                                "item_type": "str",
                                "label": "参数6",
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": "参数6，多行的文本编辑器",
                                "editable": 1,
                                "condition": "",
                                "sub_args": {}
                            },
                        }
                    },
                    "username": "admin",
                    "changed_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_on": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": "模板测试",
                    "describe": "模板测试",
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }

                }
            ],

        },
        "status": 0
    }
    @expose("/template/list/")
    def template_list(self):

        index=1
        for group in self.all_template['templte_list']:
            for template in self.all_template['templte_list'][group]:
                template['template_id']=index
                template['changed_on'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                template['created_on'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                template['username'] = 'admin'
                index+=1
        # print(json.dumps(all_template,indent=4,ensure_ascii=False))
        return jsonify(self.all_template)


    def check_pipeline_perms(user_fun):
        # @pysnooper.snoop()
        def wraps(*args, **kwargs):
            pipeline_id = int(kwargs.get('pipeline_id','0'))
            if not pipeline_id:
                response = make_response("pipeline_id not exist")
                response.status_code = 404
                return response

            user_roles = [role.name.lower() for role in g.user.roles]
            if "admin" in user_roles:
                return user_fun(*args, **kwargs)

            join_projects_id = security_manager.get_join_projects_id(db.session)
            pipeline = db.session.query(ETL_Pipeline).filter_by(id=pipeline_id).first()
            if pipeline.project.id in join_projects_id:
                return user_fun(*args, **kwargs)

            response = make_response("no perms to run pipeline %s"%pipeline_id)
            response.status_code = 403
            return response

        return wraps


    # # @event_logger.log_this
    @expose("/run_etl_pipeline/<etl_pipeline_id>", methods=["GET", "POST"])
    # @check_pipeline_perms
    def run_etl_pipeline(self,etl_pipeline_id):
        print(etl_pipeline_id)
        url = '/etl_pipeline_modelview/web/' + etl_pipeline_id
        try:
            pipeline = db.session.query(ETL_Pipeline).filter_by(id=etl_pipeline_id).first()
            # run_pipeline(pipeline)
            db.session.commit()
            url = conf.get('MODEL_URLS',{}).get('etl_task')
        except Exception as e:
            flash(str(e),category='warning')
            return self.response(400,**{"status":1,"message":str(e),"result":{}})

        return redirect(url)


    @expose("/web/<etl_pipeline_id>", methods=["GET"])
    def web(self,etl_pipeline_id):
        etl_pipeline = db.session.query(ETL_Pipeline).filter_by(id=etl_pipeline_id).first()

        # pipeline.dag_json = pipeline.fix_dag_json()
        # pipeline.expand = json.dumps(pipeline.fix_expand(), indent=4, ensure_ascii=False)

        # db_tasks = pipeline.get_tasks(db.session)
        # if db_tasks:
        #     try:
        #         tasks={}
        #         for task in db_tasks:
        #             tasks[task.name]=task.to_json()
        #         expand = core.fix_task_position(pipeline.to_json(),tasks)
        #         pipeline.expand=json.dumps(expand,indent=4,ensure_ascii=False)
        #         db.session.commit()
        #     except Exception as e:
        #         print(e)

        # db.session.commit()
        print(etl_pipeline_id)
        url = '/static/appbuilder/visonPlus/index.html?pipeline_id=%s'%etl_pipeline_id  # 前后端集成完毕，这里需要修改掉
        data = {
            "url": url
        }
        return redirect('/frontend/showOutLink?url=%s' % urllib.parse.quote(url, safe=""))
        # 返回模板
        # return self.render_template('link.html', data=data)


    # @pysnooper.snoop(watch_explode=())
    def copy_db(self,pipeline):
        new_pipeline = pipeline.clone()
        new_pipeline.name = new_pipeline.name.replace('_', '-') + "-" + uuid.uuid4().hex[:4]
        new_pipeline.created_on = datetime.datetime.now()
        new_pipeline.changed_on = datetime.datetime.now()


        # 删除其中的每个etl_task_id和task_id
        dag_json = json.loads(pipeline.dag_json) if pipeline.dag_json else {}
        dag_json_new ={}
        for task_name in dag_json:
            new_task_name = task_name
            new_task_name=new_task_name[:new_task_name.rindex('-')+1]+str(int(round(time.time() * 1000)))  # 名称变化
            dag_json_new[new_task_name]=copy.deepcopy(dag_json[task_name])
            dag_json_new[new_task_name]['etl_task_id']=''   # 去除 us 任务  id

            dag_json_new[new_task_name]['templte_common_ui_config']=self.all_template['templte_common_ui_config']

            # if 'task_id' in dag_json_new[new_task_name]:
            #     del dag_json_new[new_task_name]['task_id']
            dag_json[task_name]['new_task_name'] = new_task_name   # 记录一下之前的名称

        # print(json.dumps(dag_json_new,indent=4,ensure_ascii=False))
        # 修正上下游
        for new_task_name in dag_json_new:
            upstreams = dag_json_new[new_task_name].get('upstream', [])
            new_upstreams=[]
            for upstream_task in upstreams:
                new_upstream_task = dag_json[upstream_task]['new_task_name']
                if new_upstream_task in dag_json_new:
                    new_upstreams.append(new_upstream_task)
            dag_json_new[new_task_name]['upstream']=new_upstreams

        # print(json.dumps(dag_json_new,indent=4,ensure_ascii=False))

        new_pipeline.dag_json = json.dumps(dag_json_new,indent=4,ensure_ascii=False)

        db.session.add(new_pipeline)
        db.session.commit()

        return new_pipeline


    # @event_logger.log_this
    @action(
        "copy", __("复制任务流"), __("复制任务流"), "fa-copy", multiple=False, single=True
    )
    # @pysnooper.snoop()
    def copy(self, pipelines):

        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        try:
            for pipeline in pipelines:
                self.copy_db(pipeline)
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            logging.error(e)
            raise e

        return redirect(request.referrer)




class ETL_Pipeline_ModelView(ETL_Pipeline_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(ETL_Pipeline)
    # base_order = ("changed_on", "desc")
    # order_columns = ['changed_on']


appbuilder.add_view_no_menu(ETL_Pipeline_ModelView)


# 添加api
class ETL_Pipeline_ModelView_Api(ETL_Pipeline_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(ETL_Pipeline)
    route_base = '/etl_pipeline_modelview/api'
    search_columns=['project','name','describe','dag_json','created_by']
    # related_views = [ETL_Task_ModelView_Api, ]

    spec_label_columns = {
        "dag_json":"全部配置"
    }

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields=add_form_query_rel_fields

    def pre_add_get(self):
        self.default_filter = {
            "created_by": g.user.id
        }

    def post_list(self, items):
        flash('此部分仅提供任务流编排能力，管理员自行对接调度Azkaban/Oozie/Airflow/argo等调度平台能力','warning')
        return items

appbuilder.add_api(ETL_Pipeline_ModelView_Api)




