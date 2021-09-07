from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface

from flask_babel import gettext as __

# 将model添加成视图，并控制在前端的显示
from myapp.models.model_job import Repository,Images,Job_Template,Task,Pipeline,Workflow,Tfjob,Xgbjob,RunHistory,Pytorchjob
from myapp.models.model_team import Project,Project_User
from flask_appbuilder.actions import action

from myapp import app, appbuilder,db,event_logger

from sqlalchemy import and_, or_, select

from myapp.utils.py import py_k8s

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

from .base import (
    api,
    BaseMyappView,
    check_ownership,
    CsvResponse,
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







class CRD_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query.order_by(self.model.create_time.desc())
        return query.filter(
            or_(
                self.model.labels.contains('"%s"'%g.user.username),
            )
        ).order_by(self.model.create_time.desc())



class Crd_ModelView_Base():

    list_columns = ['name','namespace_url','create_time','status','username','stop']
    show_columns = ['name','namespace','create_time','status','annotations_html','labels_html','spec_html','status_more_html','info_json_html']
    order_columns = ['id']

    # base_permissions = ['list','delete','show']
    # label_columns = {
    #     "annotations_html": _("Annotations"),
    #     "labels_html": _("Labels"),
    #     "name": _("名称"),
    #     "spec_html": _("Spec"),
    #     "status_more_html": _("Status"),
    #     "namespace_url":_("命名空间"),
    #     "create_time":_("创建时间"),
    #     "status": _("状态"),
    #     "username": _("关联用户"),
    #     "log": _("日志"),
    # }
    crd_name =''
    base_order = ('create_time', 'desc')
    base_filters = [["id", CRD_Filter, lambda: []]]  # 设置权限过滤器


    # list
    def base_list(self):
        k8s_client = py_k8s.K8s()
        crd_info = conf.get("CRD_INFO", {}).get(self.crd_name, {})
        if crd_info:
            crds = k8s_client.get_crd_all_namespaces(group=crd_info['group'],
                                                          version=crd_info['version'],
                                                          plural=crd_info['plural'])

            # 删除所有，注意最好id从0开始
            db.session.query(self.datamodel.obj).delete()
            # db.engine.execute("alter table %s auto_increment =0"%self.datamodel.pbj.__tablename__)
            # 添加记录
            for crd in crds:
                try:
                    labels = json.loads(crd['labels'])
                    if 'run-rtx' in labels:
                        crd['username'] = labels['run-rtx']
                    elif 'upload-rtx' in labels:
                        crd['username'] = labels['upload-rtx']
                except Exception as e:
                    logging.error(e)
                crd_model = self.datamodel.obj(**crd)
                db.session.add(crd_model)

            db.session.commit()


    # 基础批量删除
    # @pysnooper.snoop()
    def base_muldelete(self,items):
        if not items:
            abort(404)
        for item in items:
            if item:
                try:
                    labels = json.loads(item.labels) if item.labels else {}
                    kubeconfig=None
                    if 'pipeline-id' in labels:
                        pipeline = db.session.query(Pipeline).filter_by(id=int(labels['pipeline-id'])).first()
                        if pipeline:
                            kubeconfig=pipeline.project.cluster['KUBECONFIG']

                    k8s_client = py_k8s.K8s(kubeconfig)
                    crd_info = conf.get("CRD_INFO", {}).get(self.crd_name, {})
                    if crd_info:
                        crd_names = k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=item.namespace,name=item.name)
                        # db_crds = db.session.query(self.datamodel.obj).filter(self.datamodel.obj.name.in_(crd_names)).all()
                        # for db_crd in db_crds:
                        #     db_crd.status = 'Deleted'
                        # db.session.commit()
                        item.status='Deleted'
                        item.change_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        db.session.commit()

                except Exception as e:
                    flash(str(e), "danger")


    def pre_delete(self,item):
        self.base_muldelete([item])


    @expose("/stop/<crd_id>")
    def stop(self, crd_id):
        crd = db.session.query(self.datamodel.obj).filter_by(id=crd_id).first()
        self.base_muldelete([crd])
        flash('清理完成','warning')
        self.update_redirect()
        return redirect(self.get_redirect())


    @action(
        "stop_all", __("Stop"), __("Stop all Really?"), "fa-trash", single=False
    )
    def stop_all(self, items):
        self.base_muldelete(items)
        self.update_redirect()
        return redirect(self.get_redirect())

    # @event_logger.log_this
    # @expose("/list/")
    # @has_access
    # def list(self):
    #     self.base_list()
    #     widgets = self._list()
    #     res = self.render_template(
    #         self.list_template, title=self.list_title, widgets=widgets
    #     )
    #     return res

    @action(
        "muldelete", __("Delete"), __("Delete all Really?"), "fa-trash", single=False
    )
    def muldelete(self, items):
        self.base_muldelete(items)
        for item in items:
            if item:
                self._delete(item.id)
        self.update_redirect()
        return redirect(self.get_redirect())


class Workflow_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query.order_by(self.model.create_time.desc())
        return query.filter(
            or_(
                self.model.labels.contains('"%s"'%g.user.username),
            )
        ).order_by(self.model.create_time.desc())

# list正在运行的workflow
class Workflow_ModelView(Crd_ModelView_Base,MyappModelView,DeleteMixin):

    base_filters = [["id", Workflow_Filter, lambda: []]]  # 设置权限过滤器

    # 删除之前的 workflow和相关容器
    def delete_workflow(self, workflow):
        try:
            k8s_client = py_k8s.K8s(workflow.pipeline.project.cluster['KUBECONFIG'])
            k8s_client.delete_workflow(
                all_crd_info=conf.get("CRD_INFO", {}),
                namespace=workflow.namespace,
                run_id=json.loads(workflow.labels).get("run-id",'')
            )
            workflow.status='Deleted'
            workflow.change_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            db.session.commit()
        except Exception as e:
            print(e)




    @expose("/stop/<crd_id>")
    def stop(self, crd_id):
        workflow = db.session.query(self.datamodel.obj).filter_by(id=crd_id).first()
        self.delete_workflow(workflow)

        flash('清理完成','warning')
        self.update_redirect()
        return redirect(self.get_redirect())


    label_title = '运行实例'
    datamodel = SQLAInterface(Workflow)
    list_columns = ['name','project','pipeline_url', 'namespace_url', 'create_time','change_time', 'final_status','status', 'username', 'log','stop']
    crd_name = 'workflow'

appbuilder.add_view(Workflow_ModelView,"运行实例",href='/workflow_modelview/list/?_flt_2_name=&_flt_2_labels=',icon = 'fa-tasks',category = '训练')


# 添加api
class Workflow_ModelView_Api(Crd_ModelView_Base,MyappModelRestApi):

    datamodel = SQLAInterface(Workflow)
    route_base = '/workflow_modelview/api'
    list_columns = ['name', 'namespace_url', 'create_time', 'status', 'username', 'log']
    crd_name = 'workflow'

appbuilder.add_api(Workflow_ModelView_Api)


appbuilder.add_separator("训练")   # 在指定菜单栏下面的每个子菜单中间添加一个分割线的显示。


# list正在运行的tfjob
class Tfjob_ModelView(Crd_ModelView_Base,MyappModelView,DeleteMixin):
    label_title = 'tf分布式任务'
    datamodel = SQLAInterface(Tfjob)
    crd_name = 'tfjob'
    list_columns = ['name','pipeline_url','run_instance','namespace_url','create_time','status','username','stop']


appbuilder.add_view(Tfjob_ModelView,"TFjob",href="/tfjob_modelview/list/?_flt_2_name=",icon = 'fa-tasks',category = '训练')
# 添加api
class Tfjob_ModelView_Api(Crd_ModelView_Base,MyappModelRestApi):
    label_title = 'tf分布式任务'
    datamodel = SQLAInterface(Tfjob)
    route_base = '/tfjob_modelview/api'
    crd_name = 'tfjob'

appbuilder.add_api(Tfjob_ModelView_Api)


# list正在运行的xgb
class Xgbjob_ModelView(Crd_ModelView_Base,MyappModelView,DeleteMixin):
    label_title = 'xgb分布式任务'
    datamodel = SQLAInterface(Xgbjob)
    crd_name = 'xgbjob'

appbuilder.add_view(Xgbjob_ModelView,"XGBjob",href="/xgbjob_modelview/list/?_flt_2_name=",icon = 'fa-tasks',category = '训练')


# 添加api
class Xgbjob_ModelView_Api(Crd_ModelView_Base,MyappModelRestApi):
    label_title = 'xgb分布式任务'
    datamodel = SQLAInterface(Xgbjob)
    route_base = '/xgbjob_modelview/api'
    crd_name = 'xgbjob'

appbuilder.add_api(Xgbjob_ModelView_Api)


# list正在运行的pytorch
class Pytorchjob_ModelView(Crd_ModelView_Base,MyappModelView,DeleteMixin):
    label_title = 'pytorch分布式任务'
    datamodel = SQLAInterface(Pytorchjob)
    crd_name = 'pytorchjob'

appbuilder.add_view(Pytorchjob_ModelView,"Pytorchjob",href="/pytorchjob_modelview/list/?_flt_2_name=",icon = 'fa-tasks',category = '训练')


# 添加api
class Pytorchjob_ModelView_Api(Crd_ModelView_Base,MyappModelRestApi):
    label_title = 'pytorch分布式任务'
    datamodel = SQLAInterface(Pytorchjob)
    route_base = '/pytorchjob_modelview/api'
    crd_name = 'pytorchjob'

appbuilder.add_api(Pytorchjob_ModelView_Api)

