from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface

# 将model添加成视图，并控制在前端的显示
from myapp.models.model_job import Repository,Images,Job_Template,Task,Pipeline,Workflow,Tfjob,Xgbjob,RunHistory,Pytorchjob

from myapp import app, appbuilder,db,event_logger


from sqlalchemy import and_, or_, select

from .baseApi import (
    MyappModelRestApi
)

from myapp import security_manager
import kfp    # 使用自定义的就要把pip安装的删除了
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
)
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config
logging = app.logger


class RunHistory_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        pipeline_ids = security_manager.get_create_pipeline_ids(db.session)
        return query.filter(
            or_(
                self.model.pipeline_id.in_(pipeline_ids),
                # self.model.project.name.in_(['public'])
            )
        )



class RunHistory_ModelView_Base():
    label_title='定时调度历史'
    datamodel = SQLAInterface(RunHistory)
    base_order = ('id', 'desc')
    order_columns = ['id']

    list_columns = ['name', 'pipeline_url','creator','created_on','log']
    # add_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','image_pull_policy','parallelism','dag_json','global_args']
    # show_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','image_pull_policy','parallelism','global_args','dag_json_html','pipeline_file_html']
    # edit_columns = add_columns
    base_filters = [["id", RunHistory_Filter, lambda: []]]  # 设置权限过滤器


class RunHistory_ModelView(RunHistory_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(RunHistory)

appbuilder.add_view(RunHistory_ModelView,"定时调度记录",icon = 'fa-sitemap',category = '训练')




# 添加api
class RunHistory_ModelView_Api(RunHistory_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(RunHistory)
    route_base = '/runhistory_modelview/api'

appbuilder.add_api(RunHistory_ModelView_Api)



