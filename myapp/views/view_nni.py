from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.model_nni import NNI
from myapp.models.model_job import Repository
from flask_appbuilder.actions import action
import pysnooper
from flask_babel import lazy_gettext
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder, db
import os
from wtforms.validators import DataRequired, Length, Regexp
from sqlalchemy import or_
from wtforms import IntegerField, SelectField, StringField, FloatField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2ManyWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MyCodeArea, MySelectMultipleField
from myapp.views.view_team import Project_Join_Filter
import copy
from myapp.utils.py.py_k8s import K8s
from flask import (
    flash,
    g,
    Markup,
    redirect,
    request
)
from .baseApi import (
    MyappModelRestApi
)
from myapp import security_manager
import time
from .base import (
    MyappFilter,
    MyappModelView,
)
from flask_appbuilder import expose
import datetime, json

conf = app.config


class NNI_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query.order_by(self.model.id.desc())

        join_projects_id = security_manager.get_join_projects_id(db.session)
        # public_project_id =
        # logging.info(join_projects_id)
        return query.filter(
            or_(
                self.model.project_id.in_(join_projects_id),
                # self.model.project.name.in_(['public'])
            )
        ).order_by(self.model.id.desc())


class NNI_ModelView_Base():
    datamodel = SQLAInterface(NNI)
    conv = GeneralModelConverter(datamodel)
    label_title = 'nni超参搜索'
    check_redirect_list_url = conf.get('MODEL_URLS', {}).get('nni', '')

    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    base_filters = [["id", NNI_Filter, lambda: []]]
    order_columns = ['id']
    list_columns = ['project', 'describe_url', 'job_type', 'creator', 'modified', 'run', 'log']
    show_columns = ['project','created_by', 'changed_by', 'created_on', 'changed_on', 'job_type', 'name', 'namespace', 'describe',
                    'parallel_trial_count', 'max_trial_count', 'objective_type','parameters',
                    'objective_goal', 'objective_metric_name', 'objective_additional_metric_names', 'algorithm_name',
                    'algorithm_setting', 'trial_spec','job_worker_image','job_worker_command',
                    'working_dir', 'volume_mount', 'node_selector', 'image_pull_policy', 'resource_memory',
                    'resource_cpu', 'resource_gpu',
                    'experiment', 'alert_status']

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields
    edit_form_extra_fields = {}

    edit_form_extra_fields["alert_status"] = MySelectMultipleField(
        label=_(datamodel.obj.lab('alert_status')),
        widget=Select2ManyWidget(),
        # default=datamodel.obj.alert_status.default.arg,
        choices=[[x, x] for x in
                 ['Pending', 'Running', 'Succeeded', 'Failed', 'Unknown', 'Waiting', 'Terminated']],
        description="选择通知状态",
    )

    edit_form_extra_fields['name'] = StringField(
        _(datamodel.obj.lab('name')),
        description='英文名(小写字母、数字、- 组成)，最长50个字符',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired(), Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]
    )
    edit_form_extra_fields['describe'] = StringField(
        _(datamodel.obj.lab('describe')),
        description='中文描述',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['namespace'] = StringField(
        _(datamodel.obj.lab('namespace')),
        description='运行命名空间',
        widget=BS3TextFieldWidget(),
        default=datamodel.obj.namespace.default.arg,
        validators=[DataRequired()]
    )

    edit_form_extra_fields['parallel_trial_count'] = IntegerField(
        _(datamodel.obj.lab('parallel_trial_count')),
        default=datamodel.obj.parallel_trial_count.default.arg,
        description='可并行的计算实例数目',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['max_trial_count'] = IntegerField(
        _(datamodel.obj.lab('max_trial_count')),
        default=datamodel.obj.max_trial_count.default.arg,
        description='最大并行的计算实例数目',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['max_failed_trial_count'] = IntegerField(
        _(datamodel.obj.lab('max_failed_trial_count')),
        default=datamodel.obj.max_failed_trial_count.default.arg,
        description='最大失败的计算实例数目',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['objective_type'] = SelectField(
        _(datamodel.obj.lab('objective_type')),
        default=datamodel.obj.objective_type.default.arg,
        description='目标函数类型（和自己代码中对应）',
        widget=Select2Widget(),
        choices=[['maximize', 'maximize'], ['minimize', 'minimize']],
        validators=[DataRequired()]
    )

    edit_form_extra_fields['objective_goal'] = FloatField(
        _(datamodel.obj.lab('objective_goal')),
        default=datamodel.obj.objective_goal.default.arg,
        description='目标门限',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['objective_metric_name'] = StringField(
        _(datamodel.obj.lab('objective_metric_name')),
        default=NNI.objective_metric_name.default.arg,
        description='目标函数（和自己代码中对应）',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['objective_additional_metric_names'] = StringField(
        _(datamodel.obj.lab('objective_additional_metric_names')),
        default=datamodel.obj.objective_additional_metric_names.default.arg,
        description='其他目标函数（和自己代码中对应）',
        widget=BS3TextFieldWidget()
    )

    algorithm_name_choices = {
        'TPE': "TPE",
        'Random': "随机搜索",
        "Anneal": "退火算法",
        "Evolution": "进化算法",
        "SMAC": "SMAC",
        "BatchTuner": "批量调参器",
        "GridSearch": "网格搜索",
        "Hyperband": "Hyperband",
        "NetworkMorphism": "Network Morphism",
        "MetisTuner": "Metis Tuner",
        "BOHB": "BOHB Advisor",
        "GPTuner": "GP Tuner",
        "PPOTuner": "PPO Tuner",
        "PBTTuner": "PBT Tuner"
    }

    algorithm_name_choices = list(algorithm_name_choices.items())

    edit_form_extra_fields['algorithm_name'] = SelectField(
        _(datamodel.obj.lab('algorithm_name')),
        default=datamodel.obj.algorithm_name.default.arg,
        description='搜索算法',
        widget=Select2Widget(),
        choices=algorithm_name_choices,
        validators=[DataRequired()]
    )
    edit_form_extra_fields['algorithm_setting'] = StringField(
        _(datamodel.obj.lab('algorithm_setting')),
        default=datamodel.obj.algorithm_setting.default.arg,
        widget=BS3TextFieldWidget(),
        description='搜索算法配置'
    )

    edit_form_extra_fields['parameters_demo'] = StringField(
        _(datamodel.obj.lab('parameters_demo')),
        description='搜索参数示例，标准json格式，注意：所有整型、浮点型都写成字符串型',
        widget=MyCodeArea(code=core.nni_parameters_demo()),
    )
    edit_form_extra_fields['parameters'] = StringField(
        _(datamodel.obj.lab('parameters')),
        default=datamodel.obj.parameters.default.arg,
        description=Markup(('搜索参数，注意：所有整型、浮点型都写成字符串型,示例：\n'+"<pre><code>%s</code></pre>"%core.nni_parameters_demo()).replace('\n','<br>')),
        widget=MyBS3TextAreaFieldWidget(rows=10),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['node_selector'] = StringField(
        _(datamodel.obj.lab('node_selector')),
        description="部署task所在的机器(目前无需填写)",
        default=datamodel.obj.node_selector.default.arg,
        widget=BS3TextFieldWidget()
    )
    edit_form_extra_fields['working_dir'] = StringField(
        _(datamodel.obj.lab('working_dir')),
        description="代码所在目录，nni代码、配置和log都将在/mnt/${your_name}/nni/目录下进行",
        default=datamodel.obj.working_dir.default.arg,
        widget=BS3TextFieldWidget()
    )
    edit_form_extra_fields['image_pull_policy'] = SelectField(
        _(datamodel.obj.lab('image_pull_policy')),
        description="镜像拉取策略(always为总是拉取远程镜像，IfNotPresent为若本地存在则使用本地镜像)",
        widget=Select2Widget(),
        choices=[['Always', 'Always'], ['IfNotPresent', 'IfNotPresent']]
    )
    edit_form_extra_fields['volume_mount'] = StringField(
        _(datamodel.obj.lab('volume_mount')),
        description='外部挂载，格式:$pvc_name1(pvc):/$container_path1,$pvc_name2(pvc):/$container_path2',
        default=datamodel.obj.volume_mount.default.arg,
        widget=BS3TextFieldWidget()
    )
    edit_form_extra_fields['resource_memory'] = StringField(
        _(datamodel.obj.lab('resource_memory')),
        default=datamodel.obj.resource_memory.default.arg,
        description='内存的资源使用限制(每个测试实例)，示例：1G，20G',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['resource_cpu'] = StringField(
        _(datamodel.obj.lab('resource_cpu')),
        default=datamodel.obj.resource_cpu.default.arg,
        description='cpu的资源使用限制(每个测试实例)(单位：核)，示例：2', widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )

    # @pysnooper.snoop()
    def set_column(self, nni=None):
        # 对编辑进行处理
        # request_data = request.args.to_dict()
        # job_type = request_data.get('job_type', 'Job')
        # if nni:
        #     job_type = nni.job_type
        #
        # job_type_choices = ['','Job']
        # job_type_choices = [[job_type_choice,job_type_choice] for job_type_choice in job_type_choices]
        #
        # if nni:
        #     self.edit_form_extra_fields['job_type'] = SelectField(
        #         _(self.datamodel.obj.lab('job_type')),
        #         description="超参搜索的任务类型",
        #         choices=job_type_choices,
        #         widget=MySelect2Widget(extra_classes="readonly",value=job_type),
        #         validators=[DataRequired()]
        #     )
        # else:
        #     self.edit_form_extra_fields['job_type'] = SelectField(
        #         _(self.datamodel.obj.lab('job_type')),
        #         description="超参搜索的任务类型",
        #         widget=MySelect2Widget(new_web=True,value=job_type),
        #         choices=job_type_choices,
        #         validators=[DataRequired()]
        #     )

        self.edit_form_extra_fields['job_worker_image'] = StringField(
            _(self.datamodel.obj.lab('job_worker_image')),
            default=conf.get('NNI_IMAGES',''),
            description='工作节点镜像',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['job_worker_command'] = StringField(
            _(self.datamodel.obj.lab('job_worker_command')),
            default='python xx.py',
            description='工作节点启动命令',
            widget=MyBS3TextAreaFieldWidget(),
            validators=[DataRequired()]
        )

        self.edit_columns = ['project', 'name', 'describe', 'parallel_trial_count', 'max_trial_count',
                             'objective_type', 'objective_goal', 'objective_metric_name',
                             'objective_additional_metric_names',
                             'algorithm_name', 'algorithm_setting', 'parameters']
        self.edit_fieldsets = [(
            lazy_gettext('common'),
            {"fields": copy.deepcopy(self.edit_columns), "expanded": True},
        )]

        task_column = ['job_worker_image', 'working_dir', 'job_worker_command', 'resource_memory', 'resource_cpu']
        self.edit_fieldsets.append((
            lazy_gettext('task args'),
            {"fields": task_column, "expanded": True},
        ))
        for column in task_column:
            self.edit_columns.append(column)

        self.edit_fieldsets.append((
            lazy_gettext('run experiment'),
            {"fields": ['alert_status'], "expanded": True},
        ))

        self.edit_columns.append('alert_status')

        self.add_form_extra_fields = self.edit_form_extra_fields
        self.add_fieldsets = self.edit_fieldsets
        self.add_columns = self.edit_columns

    pre_add_web = set_column
    pre_update_web = set_column

    # 处理form请求
    def process_form(self, form, is_created):
        # from flask_appbuilder.forms import DynamicForm
        if 'parameters_demo' in form._fields:
            del form._fields['parameters_demo']  # 不处理这个字段


    # @pysnooper.snoop()
    def deploy_nni_service(self,nni,command):
        image_secrets = conf.get('HUBSECRET', [])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])

        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(nni.project.cluster.get('KUBECONFIG',''))
        namespace = conf.get('AUTOML_NAMESPACE','automl')
        run_id='nni-'+nni.name

        try:
            nni_deploy = k8s_client.AppsV1Api.read_namespaced_deployment(name=nni.name,namespace=namespace)
            if nni_deploy:
                print('exist nni deploy')
                k8s_client.AppsV1Api.delete_namespaced_deployment(name=nni.name,namespace=namespace)
                # return
        except Exception as e:
            print(e)


        volume_mount = nni.volume_mount+",/usr/share/zoneinfo/Asia/Shanghai(hostpath):/etc/localtime"
        labels={"app": nni.name, "user": nni.created_by.username,'run-id':run_id,'pod-type':"nni"}

        k8s_client.create_debug_pod(
            namespace=namespace,
            name=nni.name,
            labels=labels,
            command=command,
            args=None,
            volume_mount=volume_mount,
            working_dir='/mnt/%s'%nni.created_by.username,
            node_selector=nni.get_node_selector(),
            resource_memory='2G',
            resource_cpu='2',
            resource_gpu='0',
            image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
            image_pull_secrets=image_secrets,
            image=conf.get('NNI_IMAGES',json.loads(nni.job_json).get('job_worker_image')) ,
            hostAliases=conf.get('HOSTALIASES',''),
            env=None,
            privileged=False,
            accounts='nni',
            username=nni.created_by.username
                              )

        k8s_client.create_service(namespace=namespace,
                           name=nni.name,
                           username=nni.created_by.username,
                           ports=[8888],
                           selector=labels
                           )

        # 创建vs
        host = nni.project.cluster.get('HOST', request.host)
        if not host:
            host = request.host
        if ':' in host:
            host = host[:host.rindex(':')]  # 如果捕获到端口号，要去掉
        vs_json = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": nni.name,
                "namespace": namespace
            },
            "spec": {
                "gateways": [
                    "kubeflow/kubeflow-gateway"
                ],
                "hosts": [
                    "*" if core.checkip(host) else host
                ],
                "http": [
                    {
                        "match": [
                            {
                                "uri": {
                                    "prefix": "/nni/%s//" % nni.name
                                }
                            },
                            {
                                "uri": {
                                    "prefix": "/nni/%s/" % nni.name
                                }
                            }
                        ],
                        "rewrite": {
                            "uri": "/nni/%s/" % nni.name
                        },
                        "route": [
                            {
                                "destination": {
                                    "host": "%s.%s.svc.cluster.local"%(nni.name,namespace),
                                    "port": {
                                        "number": 8888
                                    }
                                }
                            }
                        ],
                        "timeout": "300s"
                    }
                ]
            }
        }
        crd_info = conf.get('CRD_INFO')['virtualservice']
        k8s_client.delete_istio_ingress(namespace=namespace, name=nni.name)

        k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, body=vs_json)

        # 删除network NetworkPolicy
        try:
            # time.sleep(2)
            k8s_client.NetworkingV1Api.delete_namespaced_network_policy(namespace=namespace,name=nni.name)
        except Exception as e:
            pass

    # 生成实验
    # @pysnooper.snoop()
    @expose('/run/<nni_id>',methods=['GET','POST'])
    # @pysnooper.snoop()
    def run(self,nni_id):
        nni = db.session.query(NNI).filter(NNI.id == nni_id).first()

        image_secrets = conf.get('HUBSECRET', [])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])
        image_secrets = str(image_secrets)

        trial_template=f'''
apiVersion: frameworkcontroller.microsoft.com/v1
kind: Framework
metadata:
  name: {nni.name}
  namespace: {nni.namespace}
spec:
  executionType: Start
  retryPolicy:
    fancyRetryPolicy: true
    maxRetryCount: 2
  taskRoles:
  - name: worker
    taskNumber: 1
    frameworkAttemptCompletionPolicy:
      minFailedTaskCount: 1
      minSucceededTaskCount: 3
    task:
      retryPolicy:
        fancyRetryPolicy: false
        maxRetryCount: 0
      podGracefulDeletionTimeoutSec: 1800
      pod:
        spec:
          restartPolicy: Never
          hostNetwork: false
          imagePullSecrets: {image_secrets}

          containers:
          - name: {nni.name}
            image: {json.loads(nni.job_json).get("job_worker_image",'')}
            command: {json.loads(nni.job_json).get("job_worker_command",'').split(' ')}
            ports:
            - containerPort: 5001
            volumeMounts:
            - name: frameworkbarrier-volume
              mountPath: /mnt/frameworkbarrier
            - name: data-volume
              mountPath: /tmp/mount  
          serviceAccountName: frameworkbarrier
          initContainers:
          - name: frameworkbarrier
            image: frameworkcontroller/frameworkbarrier
            imagePullPolicy: IfNotPresent
            volumeMounts:
            - name: frameworkbarrier-volume
              mountPath: /mnt/frameworkbarrier
          volumes:
          - name: frameworkbarrier-volume
            emptyDir: {{}}
          - name: data-volume
            hostPath:
              path: {conf.get('WORKSPACE_HOST_PATH','')}/{nni.created_by.username}/nni/{nni.name}
        '''


        controll_yaml=f'''
authorName: default
experimentName: {nni.name}
trialConcurrency: {nni.parallel_trial_count}
maxExecDuration: {nni.maxExecDuration}s
maxTrialNum: {nni.max_trial_count}
logLevel: info
logCollection: none
#choice: local, remote, pai, kubeflow
trainingServicePlatform: frameworkcontroller
searchSpacePath: /mnt/{nni.created_by.username}/nni/{nni.name}/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  builtinTunerName: {nni.algorithm_name}
trial:
  codeDir: {nni.working_dir}  
frameworkcontrollerConfig:
  namespace: {conf.get('AUTOML_NAMESPACE','automl')}
  storage: pvc
  configPath: /mnt/{nni.created_by.username}/nni/{nni.name}/trial_template.yaml  
  pvc: 
    path: "/mnt/{nni.created_by.username}/nni/{nni.name}/"
'''

        code_dir = "%s/%s/nni/%s"%(conf.get('WORKSPACE_HOST_PATH',''),nni.created_by.username,nni.name)
        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        trial_template_path = os.path.join(code_dir,'trial_template.yaml')
        file = open(trial_template_path,mode='w')
        file.write(trial_template)
        file.close()

        controll_template_path = os.path.join(code_dir, 'controll_template.yaml')
        file = open(controll_template_path, mode='w')
        file.write(controll_yaml)
        file.close()

        searchSpacePath = os.path.join(code_dir, 'search_space.json')
        file = open(searchSpacePath, mode='w')
        file.write(nni.parameters)
        file.close()

        flash('nni服务部署完成',category='success')

        # 执行启动命令
        # command = ['bash','-c','mkdir -p /nni/nni_node/static/nni/%s && cp -r /nni/nni_node/static/* /nni/nni_node/static/nni/%s/ ; nnictl create --config /mnt/%s/nni/%s/controll_template.yaml -p 8888 --foreground --url_prefix nni/%s'%(nni.name,nni.name,nni.created_by.username,nni.name,nni.name)]
        # command = ['bash', '-c','nnictl create --config /mnt/%s/nni/%s/controll_template.yaml -p 8888 --foreground' % (nni.created_by.username, nni.name)]
        command = ['bash','-c','nnictl create --config /mnt/%s/nni/%s/controll_template.yaml -p 8888 --foreground --url_prefix nni/%s'%(nni.created_by.username,nni.name,nni.name)]

        print(command)
        self.deploy_nni_service(nni,command)

        return redirect(conf.get('MODEL_URLS',{}).get('nni',''))

    # @pysnooper.snoop(watch_explode=())
    def merge_trial_spec(self,item):

        image_secrets = conf.get('HUBSECRET',[])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])

        image_secrets = [
            {
                "name": hubsecret
            } for hubsecret in image_secrets
        ]

        item.job_json={}


        item.trial_spec=core.merge_job_experiment_template(
            node_selector=item.get_node_selector(),
            volume_mount=item.volume_mount,
            image=item.job_worker_image,
            image_secrets=image_secrets,
            hostAliases=conf.get('HOSTALIASES', ''),
            workingDir=item.working_dir,
            image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
            resource_memory=item.resource_memory,
            resource_cpu=item.resource_cpu,
            command=item.job_worker_command
        )

        item.job_json = {
            "job_worker_image": item.job_worker_image,
            "job_worker_command": item.job_worker_command,
        }

        item.job_json = json.dumps(item.job_json,indent=4,ensure_ascii=False)


    # 检验参数是否有效
    # @pysnooper.snoop()
    def validate_parameters(self,parameters,algorithm):
        return parameters



    @expose("/log/<nni_id>", methods=["GET", "POST"])
    def log_task(self,nni_id):
        nni = db.session.query(NNI).filter_by(id=nni_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s = K8s(nni.project.cluster.get('KUBECONFIG',''))
        namespace = conf.get('AUTOML_NAMESPACE')
        pod = k8s.get_pods(namespace=namespace, pod_name=nni.name)
        if pod:
            pod = pod[0]
            return redirect("/k8s/web/log/%s/%s/%s" % (nni.project.cluster['NAME'],namespace, nni.name))

        flash("未检测到当前搜索正在运行的容器",category='success')
        return redirect(conf.get('MODEL_URLS',{}).get('nni',''))


    # @pysnooper.snoop()
    def pre_add(self, item):

        if item.job_type is None:
            item.job_type = 'Job'
        #     raise MyappException("Job type is mandatory")

        if not item.volume_mount:
            item.volume_mount = item.project.volume_mount

        core.validate_json(item.parameters)
        item.parameters = self.validate_parameters(item.parameters, item.algorithm_name)

        item.resource_memory=core.check_resource_memory(item.resource_memory,self.src_item_json.get('resource_memory',None) if self.src_item_json else None)
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu,self.src_item_json.get('resource_cpu',None) if self.src_item_json else None)
        self.merge_trial_spec(item)
        # self.make_experiment(item)

    def pre_update(self, item):
        self.pre_add(item)

    @action(
        "copy", __("Copy NNI Experiment"), confirmation=__('Copy NNI Experiment'), icon="fa-copy",multiple=True, single=False
    )
    def copy(self, nnis):
        if not isinstance(nnis, list):
            nnis = [nnis]
        for nni in nnis:
            new_nni = nni.clone()
            new_nni.name = new_nni.name + "-copy"
            new_nni.describe = new_nni.describe + "-copy"
            new_nni.created_on = datetime.datetime.now()
            new_nni.changed_on = datetime.datetime.now()
            db.session.add(new_nni)
            db.session.commit()

        return redirect(request.referrer)


class NNI_ModelView(NNI_ModelView_Base, MyappModelView):
    datamodel = SQLAInterface(NNI)
    conv = GeneralModelConverter(datamodel)


# 添加视图和菜单
# appbuilder.add_view(NNI_ModelView,"nni超参搜索",icon = 'fa-shopping-basket',category = '超参搜索',category_icon = 'fa-share-alt')
appbuilder.add_view_no_menu(NNI_ModelView)


# appbuilder.add_view_no_menu(NNI_ModelView)

# 添加api
class NNI_ModelView_Api(NNI_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(NNI)
    conv = GeneralModelConverter(datamodel)
    route_base = '/nni_modelview/api'
    list_columns = ['project', 'describe_url', 'creator', 'modified', 'run']
    add_columns = ['project', 'name', 'describe',
                   'parallel_trial_count', 'max_trial_count', 'objective_type',
                   'objective_goal', 'objective_metric_name', 'objective_additional_metric_names', 'algorithm_name',
                   'algorithm_setting', 'parameters', 'job_json', 'working_dir', 'node_selector',
                   'resource_memory', 'resource_cpu', 'alert_status', 'job_worker_image', 'job_worker_command']
    edit_columns = add_columns


appbuilder.add_api(NNI_ModelView_Api)
