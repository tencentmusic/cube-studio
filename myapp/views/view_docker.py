from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.forms import GeneralModelConverter
import uuid
import re
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_job import Repository,Images
from myapp.views.view_team import Project_Filter
from myapp import app, appbuilder,db,event_logger

from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from myapp.views.view_team import Project_Filter,Project_Join_Filter,filter_join_org_project
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField

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
import kfp    # 使用自定义的就要把pip安装的删除了
from werkzeug.datastructures import FileStorage
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




from myapp.models.model_docker import Docker
class Docker_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        return query.filter(self.model.created_by_fk==g.user.id)



# 定义数据库视图
class Docker_ModelView_Base():
    datamodel = SQLAInterface(Docker)
    label_title='docker'
    check_redirect_list_url = '/docker_modelview/list/'
    crd_name = 'docker'
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    conv = GeneralModelConverter(datamodel)
    base_permissions = ['can_add', 'can_delete','can_edit', 'can_list', 'can_show']  # 默认为这些
    base_order = ('changed_on', 'desc')
    base_filters = [["id", Docker_Filter, lambda: []]]  # 设置权限过滤器
    order_columns = ['id']
    add_columns=['project','describe','base_image','target_image','need_gpu','consecutive_build','expand']
    edit_columns=add_columns
    list_columns=['id','project','describe','consecutive_build','image_history','debug','save']
    add_form_extra_fields=[]
    edit_form_extra_fields=add_form_extra_fields
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields=add_form_query_rel_fields
    # @pysnooper.snoop()
    def pre_add_get(self,docker=None):

        self.add_form_extra_fields['describe'] = StringField(
            _(self.datamodel.obj.lab('describe')),
            default='',
            description="目标环境描述",
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )

        self.add_form_extra_fields['target_image']=StringField(
            _(self.datamodel.obj.lab('target_image')),
            default=conf.get('REPOSITORY_ORG')+g.user.username+":xx",
            description="目标镜像名，必须为%s%s:xxx"%(conf.get('REPOSITORY_ORG'),g.user.username),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^%s%s:"%(conf.get('REPOSITORY_ORG'),g.user.username))]
        )

        self.add_form_extra_fields['base_image'] = StringField(
            _(self.datamodel.obj.lab('base_image')),
            default='',
            description=Markup(f'基础镜像和构建方法可参考：<a href="%s">点击打开</a>'%(conf.get('HELP_URL').get('docker',''))),
            widget=BS3TextFieldWidget()
        )
        # # if g.user.is_admin():
        # self.edit_columns=['describe','base_image','target_image','need_gpu','consecutive_build']
        self.edit_form_extra_fields = self.add_form_extra_fields


    pre_update_get=pre_add_get

    def pre_add(self,item):
        image_org=conf.get('REPOSITORY_ORG')+g.user.username+":"
        if image_org not in item.target_image or item.target_image==image_org:
            flash('目标镜像名称不符合规范','warning')

        if item.expand:
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)

    def pre_update(self,item):
        self.pre_add(item)
        # 如果修改了基础镜像，就把debug中的任务删除掉
        if self.src_item_json:
            if item.base_image!=self.src_item_json.get('base_image',''):
                self.delete_pod(item.id)
                item.last_image=''
                flash('发现基础镜像更换，已帮你删除之前启动的debug容器','success')

    # @event_logger.log_this
    @expose("/debug/<docker_id>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def debug(self,docker_id):
        docker = db.session.query(Docker).filter_by(id=docker_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(conf.get('CLUSTERS').get(conf.get('ENVIRONMENT')).get('KUBECONFIG',''))
        namespace = conf.get('NOTEBOOK_NAMESPACE')
        pod_name="docker-%s-%s"%(docker.created_by.username,str(docker.id))
        pod = k8s_client.get_pods(namespace=namespace,pod_name=pod_name)
        if pod:
            pod = pod[0]
        # 有历史非运行态，直接删除
        # if pod and (pod['status']!='Running' and pod['status']!='Pending'):
        if pod and pod['status'] == 'Succeeded':
            k8s_client.delete_pods(namespace=namespace,pod_name=pod_name)
            time.sleep(2)
            pod=None


        # 没有历史或者没有运行态，直接创建
        if not pod or (pod['status']!='Running' and pod['status']!='Pending'):

            command=['sh','-c','sleep 7200 && hour=`date +%H` && while [ $hour -ge 06 ];do sleep 3600;hour=`date +%H`;done']
            hostAliases = conf.get('HOSTALIASES')

            default_volume_mount = docker.project.volume_mount
            k8s_client.create_debug_pod(namespace,
                                        name=pod_name,
                                        command=command,
                                        labels={},
                                        args=None,
                                        volume_mount=json.loads(docker.expand).get('volume_mount',default_volume_mount) if docker.expand else default_volume_mount,
                                        working_dir='/mnt/%s'%docker.created_by.username,
                                        node_selector='%s=true,train=true,org=public'%('gpu' if docker.need_gpu else 'cpu'),
                                        resource_memory=json.loads(docker.expand).get('resource_memory','8G') if docker.expand else '8G',
                                        resource_cpu=json.loads(docker.expand).get('resource_cpu','4') if docker.expand else '4',
                                        resource_gpu=json.loads(docker.expand if docker.expand else '{}').get('resource_gpu','1') if docker.need_gpu else '0',
                                        image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
                                        image_pull_secrets=conf.get('HUBSECRET',[]),
                                        image= docker.last_image if docker.last_image and docker.consecutive_build else docker.base_image,
                                        hostAliases=hostAliases,
                                        env=None,
                                        privileged=None,
                                        accounts=None,
                                        username=docker.created_by.username)

        try_num=5
        while(try_num>0):
            pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
            # print(pod)
            if pod:
                pod = pod[0]
            # 有历史非运行态，直接删除
            if pod and pod['status'] == 'Running':
                break
            try_num=try_num-1
            time.sleep(2)
        if try_num==0:
            flash('启动时间过长，一分钟后重试','warning')
            return redirect('/docker_modelview/list/')

        flash('镜像调试只安装环境，请不要运行业务代码。当晚前请注意保存镜像','warning')
        return redirect("/docker_modelview/web/debug/%s/%s/%s"%(conf.get('ENVIRONMENT'),namespace,pod_name))


    # @event_logger.log_this
    @expose("/delete_pod/<docker_id>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def delete_pod(self,docker_id):
        docker = db.session.query(Docker).filter_by(id=docker_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(conf.get('CLUSTERS').get(conf.get('ENVIRONMENT')).get('KUBECONFIG',''))
        namespace = conf.get('NOTEBOOK_NAMESPACE')
        pod_name="docker-%s-%s"%(docker.created_by.username,str(docker.id))
        k8s_client.delete_pods(namespace=namespace,pod_name=pod_name)
        flash('清理结束，可重新进行调试','success')
        return redirect("/docker_modelview/list/")


    @expose("/web/debug/<cluster_name>/<namespace>/<pod_name>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def web_debug(self,cluster_name,namespace,pod_name):
        cluster=conf.get('CLUSTERS',{})
        if cluster_name in cluster:
            pod_url = cluster[cluster_name].get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name,pod_name, namespace)
        else:
            pod_url = conf.get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name, pod_name, namespace)
        print(pod_url)
        data = {
            "url": pod_url,
            "target":'div.kd-scroll-container', #  'div.kd-scroll-container.ng-star-inserted',
            "delay": 2000,
            "loading": True,
            "currentHeight": 128
        }
        # 返回模板
        if cluster_name==conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)



    # @event_logger.log_this
    @expose("/save/<docker_id>", methods=["GET", "POST"])
    # @pysnooper.snoop(watch_explode='status')
    def save(self,docker_id):
        docker = db.session.query(Docker).filter_by(id=docker_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(conf.get('CLUSTERS').get(conf.get('ENVIRONMENT')).get('KUBECONFIG',''))
        namespace = conf.get('NOTEBOOK_NAMESPACE')
        pod_name="docker-%s-%s"%(docker.created_by.username,str(docker.id))
        pod = k8s_client.v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        node_name=''
        container_id=''
        if pod:
            node_name=pod.spec.node_name
            containers = [container for container in pod.status.container_statuses if container.name==pod_name]
            if containers:
                container_id = containers[0].container_id.replace('docker://','')

        if not node_name or not container_id:
            flash('没有发现正在运行的调试镜像，请先调试惊险，安装环境后，再保存生成新镜像',category='warning')
            return redirect('/docker_modelview/list/')

        # flash('新镜像正在保存推送中，请留意消息通知',category='success')
        # return redirect('/docker_modelview/list/')

        pod_name = "docker-commit-%s-%s" % (docker.created_by.username, str(docker.id))
        command = ['sh', '-c', 'docker commit %s %s && docker push %s'%(container_id,docker.target_image,docker.target_image)]
        hostAliases = conf.get('HOSTALIASES')
        k8s_client.create_debug_pod(
            namespace=namespace,
            name=pod_name,
            command=command,
            labels={},
            args=None,
            volume_mount='/var/run/docker.sock(hostpath):/var/run/docker.sock',
            working_dir='/mnt/%s' % docker.created_by.username,
            node_selector=None,
            resource_memory='4G',
            resource_cpu='4',
            resource_gpu='0',
            image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
            image_pull_secrets=conf.get('HUBSECRET', []),
            image='ccr.ccs.tencentyun.com/cube-studio/docker',
            hostAliases=hostAliases,
            env=None,
            privileged=None,
            accounts=None,
            username=docker.created_by.username,
            node_name=node_name
        )
        from myapp.tasks.async_task import check_docker_commit
        # 发起异步任务检查commit pod是否完成，如果完成，修正last_image
        kwargs={
            "docker_id":docker.id
        }
        check_docker_commit.apply_async(kwargs=kwargs)

        return redirect("/myapp/web/log/%s/%s/%s" % (conf.get('ENVIRONMENT'),namespace, pod_name))




class Docker_ModelView(Docker_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Docker)

# 添加视图和菜单
appbuilder.add_view(Docker_ModelView,"镜像调试",href="/docker_modelview/list/",icon = 'fa-cubes',category = '在线开发',category_icon = 'fa-glass')









