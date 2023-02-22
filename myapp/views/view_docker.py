from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import lazy_gettext as _
from flask_appbuilder.forms import GeneralModelConverter
from wtforms.validators import DataRequired
from myapp.models.model_job import Repository
from myapp import app, appbuilder,db

from wtforms import StringField
from myapp.views.view_team import Project_Join_Filter
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from myapp.forms import MyBS3TextAreaFieldWidget

from .baseApi import MyappModelRestApi
from flask import (
    flash,
    g,
    Markup,
    redirect
)
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
from flask_appbuilder import expose
import datetime,time,json
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



class Docker_ModelView_Base():
    datamodel = SQLAInterface(Docker)
    label_title='docker'
    check_redirect_list_url = conf.get('MODEL_URLS',{}).get('docker')

    crd_name = 'docker'

    conv = GeneralModelConverter(datamodel)
    base_permissions = ['can_add', 'can_delete','can_edit', 'can_list', 'can_show']
    base_order = ('changed_on', 'desc')
    base_filters = [["id", Docker_Filter, lambda: []]]
    order_columns = ['id']
    add_columns=['project','describe','base_image','target_image','need_gpu','consecutive_build','expand']
    edit_columns=add_columns
    list_columns=['project','describe','consecutive_build','image_history','debug']
    cols_width={
        "project": {"type": "ellip2", "width": 200},
        "describe":{"type": "ellip2", "width": 200},
        "image_history":{"type": "ellip3", "width": 700},
        "debug":{"type": "ellip2", "width": 300}
    }

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields=add_form_query_rel_fields
    expand={
        "volume_mount":"kubeflow-user-workspace(pvc):/mnt",
        "resource_memory":"8G",
        "resource_cpu": "4"
    }
    add_form_extra_fields={
        "describe":StringField(
            _(datamodel.obj.lab('describe')),
            default='',
            description="目标镜像描述",
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "base_image":StringField(
            _(datamodel.obj.lab('base_image')),
            default='',
            description=Markup('基础镜像和构建方法可参考：<a href="%s">点击打开</a>'%(conf.get('HELP_URL').get('docker',''))),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),]
        ),
        "expand":StringField(
            _(datamodel.obj.lab('expand')),
            default=json.dumps(expand,ensure_ascii=False,indent=4),
            description=Markup('扩展字段'),
            widget=MyBS3TextAreaFieldWidget(rows=3)
        )

    }
    edit_form_extra_fields=add_form_extra_fields
    # @pysnooper.snoop()
    def pre_add_web(self,docker=None):
        self.add_form_extra_fields['target_image']=StringField(
            _(self.datamodel.obj.lab('target_image')),
            default=conf.get('REPOSITORY_ORG')+g.user.username+":"+datetime.datetime.now().strftime('%Y.%m.%d'+".1"),
            description="目标镜像名，将直接推送到目标仓库，需在镜像仓库中配置了相应仓库的账号密码",
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),]
        )
        # # if g.user.is_admin():
        # self.edit_columns=['describe','base_image','target_image','need_gpu','consecutive_build']
        self.edit_form_extra_fields = self.add_form_extra_fields


    pre_update_web=pre_add_web

    def pre_add(self,item):
        image_org=conf.get('REPOSITORY_ORG')+g.user.username+":"
        if image_org not in item.target_image or item.target_image==image_org:
            flash('目标镜像名称不符合规范','warning')

        if item.expand:
            expand = json.loads(item.expand)
            expand['namespace'] = json.loads(item.project.expand).get('NOTEBOOK_NAMESPACE', conf.get('NOTEBOOK_NAMESPACE'))
            item.expand = json.dumps(expand,indent=4,ensure_ascii=False)



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
        namespace = json.loads(docker.expand).get("namespace",conf.get('NOTEBOOK_NAMESPACE'))
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
                                        labels={"app":"docker","user":g.user.username,"pod-type":"docker"},
                                        args=None,
                                        volume_mount=json.loads(docker.expand).get('volume_mount',default_volume_mount) if docker.expand else default_volume_mount,
                                        working_dir='/mnt/%s'%docker.created_by.username,
                                        node_selector='%s=true,train=true,org=%s'%('gpu' if docker.need_gpu else 'cpu',docker.project.org),
                                        resource_memory="0~"+json.loads(docker.expand).get('resource_memory','8G') if docker.expand else '8G',
                                        resource_cpu='0~'+json.loads(docker.expand).get('resource_cpu','4') if docker.expand else '4',
                                        resource_gpu=json.loads(docker.expand if docker.expand else '{}').get('resource_gpu','1') if docker.need_gpu else '0',
                                        image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
                                        image_pull_secrets=conf.get('HUBSECRET',[]),
                                        image= docker.last_image if docker.last_image and docker.consecutive_build else docker.base_image,
                                        hostAliases=hostAliases,
                                        env={
                                            "USERNAME": docker.created_by.username
                                        },
                                        privileged=None,
                                        accounts=None,
                                        username=docker.created_by.username)

        try_num=20
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
            message='拉取镜像时间过长，一分钟后刷新此页面'
            flash(message,'warning')
            return self.response(400,**{"message":message,"status":1,"result":pod['status_more']})
            # return redirect(conf.get('MODEL_URLS',{}).get('docker',''))

        flash('镜像调试只安装环境，请不要运行业务代码。当晚前请注意保存镜像','warning')
        return redirect("/docker_modelview/web/debug/%s/%s/%s"%(conf.get('ENVIRONMENT'),namespace,pod_name))


    # @event_logger.log_this
    @expose("/delete_pod/<docker_id>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def delete_pod(self,docker_id):
        docker = db.session.query(Docker).filter_by(id=docker_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(conf.get('CLUSTERS').get(conf.get('ENVIRONMENT')).get('KUBECONFIG',''))
        namespace = json.loads(docker.expand).get("namespace",conf.get('NOTEBOOK_NAMESPACE'))
        pod_name="docker-%s-%s"%(docker.created_by.username,str(docker.id))
        k8s_client.delete_pods(namespace=namespace,pod_name=pod_name)
        pod_name="docker-commit-%s-%s"%(docker.created_by.username,str(docker.id))
        k8s_client.delete_pods(namespace=namespace,pod_name=pod_name)
        flash('清理结束，可重新进行调试','success')
        return redirect(conf.get('MODEL_URLS',{}).get('docker',''))


    @expose("/web/debug/<cluster_name>/<namespace>/<pod_name>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def web_debug(self,cluster_name,namespace,pod_name):
        from flask import request
        clusters=conf.get('CLUSTERS',{})
        if cluster_name in clusters:
            pod_url = "http://" + clusters[cluster_name].get('HOST', request.host) + conf.get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name,pod_name, namespace)
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
        namespace = json.loads(docker.expand).get("namespace",conf.get('NOTEBOOK_NAMESPACE'))
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
            message = '没有发现正在运行的调试镜像，请先调试惊险，安装环境后，再保存生成新镜像'
            flash(message,category='warning')
            return self.response(400,**{"message":message,"status":1,"result":{}})
            # return redirect(conf.get('MODEL_URLS',{}).get('docker',''))

        # flash('新镜像正在保存推送中，请留意消息通知',category='success')
        # return redirect(conf.get('MODEL_URLS',{}).get('docker',''))

        pod_name = "docker-commit-%s-%s" % (docker.created_by.username, str(docker.id))
        login_command=''
        all_repositorys = db.session.query(Repository).all()
        for repo in all_repositorys:
            if repo.server in docker.target_image:
                login_command = 'docker login --username %s --password %s %s'%(repo.user,repo.password,repo.server)
        if login_command:
            command = ['sh', '-c', 'docker commit %s %s && %s && docker push %s'%(container_id,docker.target_image,login_command,docker.target_image)]
        else:
            command = ['sh', '-c', 'docker commit %s %s && docker push %s'%(container_id,docker.target_image,docker.target_image)]

        hostAliases = conf.get('HOSTALIASES')
        k8s_client.create_debug_pod(
            namespace=namespace,
            name=pod_name,
            command=command,
            labels={"app":"docker","user":g.user.username,"pod-type":"docker"},
            args=None,
            volume_mount='/var/run/docker.sock(hostpath):/var/run/docker.sock',
            working_dir='/mnt/%s' % docker.created_by.username,
            node_selector=None,
            resource_memory='0~4G',
            resource_cpu='0~4',
            resource_gpu='0',
            image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
            image_pull_secrets=conf.get('HUBSECRET', []),
            image=conf.get('DOCKER_IMAGE','ccr.ccs.tencentyun.com/cube-studio/docker'),
            hostAliases=hostAliases,
            env={
                "USERNAME": docker.created_by.username
            },
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

        return redirect("/k8s/web/log/%s/%s/%s" % (conf.get('ENVIRONMENT'),namespace, pod_name))




class Docker_ModelView(Docker_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Docker)

appbuilder.add_view_no_menu(Docker_ModelView)

# 添加api
class Docker_ModelView_Api(Docker_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Docker)
    route_base = '/docker_modelview/api'
    add_columns = ['project', 'describe', 'base_image', 'target_image', 'need_gpu', 'consecutive_build', 'expand']
    edit_columns = add_columns
    search_columns = ['created_by','project']

appbuilder.add_api(Docker_ModelView_Api)







