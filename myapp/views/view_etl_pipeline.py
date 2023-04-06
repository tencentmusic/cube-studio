from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import pysnooper
import urllib.parse
from sqlalchemy.exc import InvalidRequestError
import importlib

from myapp.models.model_etl_pipeline import ETL_Pipeline,ETL_Task
from myapp.views.view_team import Project_Join_Filter
from flask_appbuilder.actions import action
from flask import jsonify
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms.validators import DataRequired, Length, Regexp
from sqlalchemy import or_
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget
import copy
from .baseApi import MyappModelRestApi
from flask import (
    flash,
    g,
    make_response,
    redirect,
    request,
)
from myapp import security_manager
from myapp.views.view_team import filter_join_org_project

from .base import (
    DeleteMixin,
    get_user_roles,
    MyappFilter,
    MyappModelView,
)

from flask_appbuilder import expose
import datetime,time,json
conf = app.config
logging = app.logger

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
    def pre_add_web(self):
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

    list_columns = ['id','project','etl_pipeline_url','workflow','creator','modified']
    cols_width = {
        "project":{"type": "ellip2", "width": 200},
        "etl_pipeline_url": {"type": "ellip2", "width": 400},
        "creator": {"type": "ellip2", "width": 100},
        "modified": {"type": "ellip2", "width": 100},
    }


    add_columns = ['project','name','describe','workflow']
    show_columns = ['project','name','describe','config','dag_json','created_by','changed_by','created_on','changed_on','expand','workflow']
    edit_columns = add_columns


    base_filters = [["id", ETL_Pipeline_Filter, lambda: []]]
    conv = GeneralModelConverter(datamodel)

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
        ),
        "workflow": SelectField(
            _('workflow'),
            widget=MySelect2Widget(),
            default='airflow',
            description='调度集群选择',
            choices=[['airflow', 'airflow'],['dophinscheduler','dophinscheduler'],['azkaban','azkaban']],
            validators=[DataRequired()]
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
                if 'templte_ui_config' in dag_json[task_name]:
                    del dag_json[task_name]['templte_ui_config']

            for node_name in dag_json:
                if not dag_json[node_name].get('task_id',''):
                    dag_json[node_name]['task_id']=uuid.uuid4().hex[:6]
            item.dag_json = json.dumps(dag_json,indent=4,ensure_ascii=False)


    # 删除前先把下面的task删除了
    # @pysnooper.snoop()
    def pre_delete(self, pipeline):
        params = importlib.import_module('myapp.views.view_etl_pipeline_' + pipeline.workflow)
        etl_pipeline = getattr(params, pipeline.workflow.upper() + '_ETL_PIPELINE')(pipeline)
        etl_pipeline.delete_pipeline()

        # 删除本地task
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

    # 获取pipeline配置信息，包括快捷菜单，运行按钮，公共配置参数，任务流dag_json
    @expose("/config/<etl_pipeline_id>",methods=("GET",'POST'))
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

        params = importlib.import_module('myapp.views.view_etl_pipeline_' + pipeline.workflow)
        etl_pipeline = getattr(params, pipeline.workflow.upper() + '_ETL_PIPELINE')(pipeline)

        config = {
            "id":pipeline.id,
            "name":pipeline.name,
            "label":pipeline.describe,
            "project":pipeline.project.describe,
            "pipeline_ui_config":etl_pipeline.pipeline_config_ui,
            "pipeline_jump_button": etl_pipeline.pipeline_jump_button,
            "pipeline_run_button": etl_pipeline.pipeline_run_button,
            "dag_json":back_dag_json,
            "config": json.loads(pipeline.config),
            "message": "success",
            "status": 0
        }

        return jsonify(config)


    @expose("/template/list/<etl_pipeline_id>")
    # @pysnooper.snoop()
    def template_list(self,etl_pipeline_id):
        pipeline = db.session.query(ETL_Pipeline).filter_by(id=etl_pipeline_id).first()
        params = importlib.import_module('myapp.views.view_etl_pipeline_'+pipeline.workflow)
        etl_pipeline = getattr(params, pipeline.workflow.upper() + '_ETL_PIPELINE')(pipeline)
        all_template = etl_pipeline.all_template

        index=1
        for group in all_template['templte_list']:
            for template in all_template['templte_list'][group]:
                template['template_id']=index
                template['changed_on'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                template['created_on'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                template['username'] = 'admin'
                index+=1
        # print(json.dumps(all_template,indent=4,ensure_ascii=False))
        return jsonify(all_template)


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
    @expose("/submit_etl_pipeline/<etl_pipeline_id>", methods=["GET", "POST"])
    def submit_etl_pipeline(self,etl_pipeline_id):
        print(etl_pipeline_id)
        url = '/etl_pipeline_modelview/web/' + etl_pipeline_id
        try:
            pipeline = db.session.query(ETL_Pipeline).filter_by(id=etl_pipeline_id).first()
            params = importlib.import_module('myapp.views.view_etl_pipeline_' + pipeline.workflow)
            etl_pipeline = getattr(params, pipeline.workflow.upper() + '_ETL_PIPELINE')(pipeline)
            dag_json, redirect_url = etl_pipeline.submit_pipeline()
            if dag_json:
                if type(dag_json)==dict:
                    dag_json=json.dumps(dag_json,indent=4,ensure_ascii=False)
                pipeline.dag_json = dag_json
                db.session.commit()

            if redirect_url:
                return redirect(redirect_url)
        except Exception as e:
            flash(str(e),category='warning')
            import traceback
            return self.response(400,**{"status":1,"message":traceback.format_exc(),"result":{}})

        url = conf.get('MODEL_URLS', {}).get('etl_task')
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

            # dag_json_new[new_task_name]['templte_common_ui_config']=self.all_template['templte_common_ui_config']

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
        "copy_pipeline", __("复制任务流"), __("复制任务流"), "fa-copy", multiple=False, single=True
    )
    # @pysnooper.snoop()
    def copy_pipeline(self, pipelines):

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

appbuilder.add_view_no_menu(ETL_Pipeline_ModelView)

# 添加api
class ETL_Pipeline_ModelView_Api(ETL_Pipeline_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(ETL_Pipeline)
    route_base = '/etl_pipeline_modelview/api'
    search_columns=['id','project','name','describe','dag_json','created_by']
    # related_views = [ETL_Task_ModelView_Api, ]

    spec_label_columns = {
        "dag_json":"全部配置",
        "workflow": "调度引擎"
    }

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields=add_form_query_rel_fields

    def pre_add_web(self):
        self.default_filter = {
            "created_by": g.user.id
        }

    def post_list(self, items):
        flash('此部分仅提供任务流编排能力，管理员自行对接调度Azkaban/Oozie/Airflow/argo等调度平台能力','warning')
        return items

appbuilder.add_api(ETL_Pipeline_ModelView_Api)




