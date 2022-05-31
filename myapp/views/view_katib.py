from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
# 将model添加成视图，并控制在前端的显示
import uuid
from myapp.models.model_katib import Hyperparameter_Tuning
from myapp.models.model_job import Repository
from flask_appbuilder.actions import action

from flask_appbuilder.models.sqla.filters import FilterEqualFunction, FilterStartsWith,FilterEqual,FilterNotEqual
from wtforms.validators import EqualTo,Length
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db,event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
import os,sys
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from sqlalchemy import and_, or_, select
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField, SelectField, StringField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MyCommaSeparatedListField,MySelectMultipleField
from myapp.views.view_team import Project_Filter
from myapp.utils.py import py_k8s
from flask_wtf.file import FileField
import shlex
import re,copy
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
from .baseApi import (
    MyappModelRestApi
)
from myapp import security_manager

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

from kubernetes.client import V1ObjectMeta
import kubeflow.katib as kc
from kubeflow.katib import constants
from kubeflow.katib import utils
from kubeflow.katib import V1alpha3AlgorithmSetting
from kubeflow.katib import V1alpha3AlgorithmSetting
from kubeflow.katib import V1alpha3AlgorithmSpec
from kubeflow.katib import V1alpha3CollectorSpec
from kubeflow.katib import V1alpha3EarlyStoppingSetting
from kubeflow.katib import V1alpha3EarlyStoppingSpec
from kubeflow.katib import V1alpha3Experiment
from kubeflow.katib import V1alpha3ExperimentCondition
from kubeflow.katib import V1alpha3ExperimentList
from kubeflow.katib import V1alpha3ExperimentSpec
from kubeflow.katib import V1alpha3ExperimentStatus
from kubeflow.katib import V1alpha3FeasibleSpace
from kubeflow.katib import V1alpha3FileSystemPath
from kubeflow.katib import V1alpha3FilterSpec
from kubeflow.katib import V1alpha3GoTemplate
from kubeflow.katib import V1alpha3GraphConfig
from kubeflow.katib import V1alpha3Metric
from kubeflow.katib import V1alpha3MetricsCollectorSpec
from kubeflow.katib import V1alpha3NasConfig
from kubeflow.katib import V1alpha3ObjectiveSpec
from kubeflow.katib import V1alpha3Observation
from kubeflow.katib import V1alpha3Operation
from kubeflow.katib import V1alpha3OptimalTrial
from kubeflow.katib import V1alpha3ParameterAssignment
from kubeflow.katib import V1alpha3ParameterSpec
from kubeflow.katib import V1alpha3SourceSpec
from kubeflow.katib import V1alpha3Suggestion
from kubeflow.katib import V1alpha3SuggestionCondition
from kubeflow.katib import V1alpha3SuggestionList
from kubeflow.katib import V1alpha3SuggestionSpec
from kubeflow.katib import V1alpha3SuggestionStatus
from kubeflow.katib import V1alpha3TemplateSpec
from kubeflow.katib import V1alpha3Trial
from kubeflow.katib import V1alpha3TrialAssignment
from kubeflow.katib import V1alpha3TrialCondition
from kubeflow.katib import V1alpha3TrialList
from kubeflow.katib import V1alpha3TrialSpec
from kubeflow.katib import V1alpha3TrialStatus
from kubeflow.katib import V1alpha3TrialTemplate


conf = app.config

class HP_Filter(MyappFilter):
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


# 定义数据库视图
class Hyperparameter_Tuning_ModelView_Base():
    datamodel = SQLAInterface(Hyperparameter_Tuning)
    conv = GeneralModelConverter(datamodel)
    label_title='超参搜索'
    check_redirect_list_url = '/hyperparameter_tuning_modelview/list/'
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''


    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']  # 默认为这些
    base_order = ('id', 'desc')
    base_filters = [["id", HP_Filter, lambda: []]]  # 设置权限过滤器
    order_columns = ['id']
    list_columns = ['project','name_url','describe','job_type','creator','run_url','modified']
    show_columns = ['created_by','changed_by','created_on','changed_on','job_type','name','namespace','describe',
                    'parallel_trial_count','max_trial_count','max_failed_trial_count','objective_type',
                    'objective_goal','objective_metric_name','objective_additional_metric_names','algorithm_name',
                    'algorithm_setting','parameters_html','trial_spec_html','experiment_html']


    add_form_query_rel_fields = {
        "project": [["name", Project_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields
    edit_form_extra_fields={}

    edit_form_extra_fields["alert_status"] = MySelectMultipleField(
        label=_(datamodel.obj.lab('alert_status')),
        widget=Select2ManyWidget(),
        choices=[[x, x] for x in
                 ['Pending', 'Running', 'Succeeded', 'Failed', 'Unknown', 'Waiting', 'Terminated']],
        description="选择通知状态",
    )

    edit_form_extra_fields['name'] = StringField(
        _(datamodel.obj.lab('name')),
        description='英文名(字母、数字、- 组成)，最长50个字符',
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
        default=datamodel.obj.objective_metric_name.default.arg,
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
    algorithm_name_choices = ['grid', 'random', 'hyperband', 'bayesianoptimization']
    algorithm_name_choices = [[algorithm_name_choice, algorithm_name_choice] for algorithm_name_choice in
                              algorithm_name_choices]
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
        widget=MyCodeArea(code=core.hp_parameters_demo()),
    )
    edit_form_extra_fields['parameters'] = StringField(
        _(datamodel.obj.lab('parameters')),
        default=datamodel.obj.parameters.default.arg,
        description='搜索参数，注意：所有整型、浮点型都写成字符串型',
        widget=MyBS3TextAreaFieldWidget(rows=10),
        validators=[DataRequired()]
    )
    edit_form_extra_fields['node_selector'] = StringField(
        _(datamodel.obj.lab('node_selector')),
        description="部署task所在的机器(目前无需填写)",
        widget=BS3TextFieldWidget()
    )
    edit_form_extra_fields['working_dir'] = StringField(
        _(datamodel.obj.lab('working_dir')),
        description="工作目录，如果为空，则使用Dockerfile中定义的workingdir",
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
    def set_column(self, hp=None):
        # 对编辑进行处理
        request_data = request.args.to_dict()
        job_type = request_data.get('job_type', '')
        if hp:
            job_type = hp.job_type

        job_type_choices = ['','TFJob','XGBoostJob','PyTorchJob','Job']
        job_type_choices = [[job_type_choice,job_type_choice] for job_type_choice in job_type_choices]

        if hp:
            self.edit_form_extra_fields['job_type'] = SelectField(
                _(self.datamodel.obj.lab('job_type')),
                description="超参搜索的任务类型",
                choices=job_type_choices,
                widget=MySelect2Widget(extra_classes="readonly",value=job_type),
                validators=[DataRequired()]
            )
        else:
            self.edit_form_extra_fields['job_type'] = SelectField(
                _(self.datamodel.obj.lab('job_type')),
                description="超参搜索的任务类型",
                widget=MySelect2Widget(new_web=True,value=job_type),
                choices=job_type_choices,
                validators=[DataRequired()]
            )


        self.edit_form_extra_fields['tf_worker_num'] = IntegerField(
            _(self.datamodel.obj.lab('tf_worker_num')),
            default=json.loads(hp.job_json).get('tf_worker_num',3) if hp and hp.job_json else 3,
            description='工作节点数目',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['tf_worker_image'] = StringField(
            _(self.datamodel.obj.lab('tf_worker_image')),
            default=json.loads(hp.job_json).get('tf_worker_image',conf.get('KATIB_TFJOB_DEFAULT_IMAGE','')) if hp and hp.job_json else conf.get('KATIB_TFJOB_DEFAULT_IMAGE',''),
            description='工作节点镜像',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['tf_worker_command'] = StringField(
            _(self.datamodel.obj.lab('tf_worker_command')),
            default=json.loads(hp.job_json).get('tf_worker_command','python xx.py') if hp and hp.job_json else 'python xx.py',
            description='工作节点启动命令',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['job_worker_image'] = StringField(
            _(self.datamodel.obj.lab('job_worker_image')),
            default=json.loads(hp.job_json).get('job_worker_image',conf.get('KATIB_JOB_DEFAULT_IMAGE','')) if hp and hp.job_json else conf.get('KATIB_JOB_DEFAULT_IMAGE',''),
            description='工作节点镜像',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['job_worker_command'] = StringField(
            _(self.datamodel.obj.lab('job_worker_command')),
            default=json.loads(hp.job_json).get('job_worker_command','python xx.py') if hp and hp.job_json else 'python xx.py',
            description='工作节点启动命令',
            widget=MyBS3TextAreaFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['pytorch_worker_num'] = IntegerField(
            _(self.datamodel.obj.lab('pytorch_worker_num')),
            default=json.loads(hp.job_json).get('pytorch_worker_num', 3) if hp and hp.job_json else 3,
            description='工作节点数目',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['pytorch_worker_image'] = StringField(
            _(self.datamodel.obj.lab('pytorch_worker_image')),
            default=json.loads(hp.job_json).get('pytorch_worker_image',conf.get('KATIB_PYTORCHJOB_DEFAULT_IMAGE','')) if hp and hp.job_json else conf.get('KATIB_PYTORCHJOB_DEFAULT_IMAGE',''),
            description='工作节点镜像',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['pytorch_master_command'] = StringField(
            _(self.datamodel.obj.lab('pytorch_master_command')),
            default=json.loads(hp.job_json).get('pytorch_master_command',
                                                'python xx.py') if hp and hp.job_json else 'python xx.py',
            description='master节点启动命令',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.edit_form_extra_fields['pytorch_worker_command'] = StringField(
            _(self.datamodel.obj.lab('pytorch_worker_command')),
            default=json.loads(hp.job_json).get('pytorch_worker_command',
                                                'python xx.py') if hp and hp.job_json else 'python xx.py',
            description='工作节点启动命令',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )

        self.edit_columns = ['job_type','project','name','namespace','describe','parallel_trial_count','max_trial_count','max_failed_trial_count',
                          'objective_type','objective_goal','objective_metric_name','objective_additional_metric_names',
                          'algorithm_name','algorithm_setting','parameters_demo',
                          'parameters']
        self.edit_fieldsets=[(
            lazy_gettext('common'),
            {"fields":  copy.deepcopy(self.edit_columns), "expanded": True},
        )]

        if job_type=='TFJob':
            group_columns = ['tf_worker_num','tf_worker_image','tf_worker_command']
            self.edit_fieldsets.append((
                lazy_gettext(job_type),
                {"fields":group_columns, "expanded": True},
            )
            )
            for column in group_columns:
                self.edit_columns.append(column)
        if job_type=='Job':
            group_columns = ['job_worker_image','job_worker_command']
            self.edit_fieldsets.append((
                lazy_gettext(job_type),
                {"fields":group_columns, "expanded": True},
            )
            )
            for column in group_columns:
                self.edit_columns.append(column)
        if job_type=='PyTorchJob':
            group_columns = ['pytorch_worker_num','pytorch_worker_image','pytorch_master_command','pytorch_worker_command']
            self.edit_fieldsets.append((
                lazy_gettext(job_type),
                {"fields":group_columns, "expanded": True},
            )
            )
            for column in group_columns:
                self.edit_columns.append(column)

        if job_type=='XGBoostJob':
            group_columns = ['pytorchjob_worker_image','pytorchjob_worker_command']
            self.edit_fieldsets.append((
                lazy_gettext(job_type),
                {"fields":group_columns, "expanded": True},
            )
            )
            for column in group_columns:
                self.edit_columns.append(column)


        task_column=['working_dir','volume_mount','node_selector','image_pull_policy','resource_memory','resource_cpu']
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
        self.add_columns=self.edit_columns


    # 处理form请求
    def process_form(self, form, is_created):
        # from flask_appbuilder.forms import DynamicForm
        if 'parameters_demo' in form._fields:
            del form._fields['parameters_demo']  # 不处理这个字段

    # 生成实验
    # @pysnooper.snoop()
    def make_experiment(self,item):

        # 搜索算法相关
        algorithmsettings = []
        for setting in item.algorithm_setting.strip().split(','):
            setting = setting.strip()
            if setting:
                key,value = setting.split('=')[0].strip(),setting.split('=')[1].strip()
                algorithmsettings.append(V1alpha3AlgorithmSetting(name=key,value=value))

        algorithm = V1alpha3AlgorithmSpec(
            algorithm_name=item.algorithm_name,
            algorithm_settings=algorithmsettings if algorithmsettings else None
        )

        # 实验结果度量，很多中搜集方式，这里不应该写死这个。
        metrics_collector_spec=None
        if item.job_type=='TFJob':
            collector = V1alpha3CollectorSpec(kind="TensorFlowEvent")
            source = V1alpha3SourceSpec(V1alpha3FileSystemPath(kind="Directory", path="/train"))
            metrics_collector_spec = V1alpha3MetricsCollectorSpec(
                collector=collector,
                source=source)
        elif item.job_type=='Job':
            pass


        # 目标函数
        objective = V1alpha3ObjectiveSpec(
            goal=item.objective_goal,
            objective_metric_name=item.objective_metric_name,
            type=item.objective_type)

        # 搜索参数
        parameters=[]
        hp_parameters = json.loads(item.parameters)
        for parameter in hp_parameters:
            if hp_parameters[parameter]['type']=='int' or hp_parameters[parameter]['type']=='double':
                feasible_space = V1alpha3FeasibleSpace(
                    min=str(hp_parameters[parameter]['min']),
                    max=str(hp_parameters[parameter]['max']),
                    step = str(hp_parameters[parameter].get('step','')) if hp_parameters[parameter].get('step','') else None)
                parameters.append(V1alpha3ParameterSpec(
                    feasible_space=feasible_space,
                    name=parameter,
                    parameter_type=hp_parameters[parameter]['type']
                ))
            elif hp_parameters[parameter]['type']=='categorical':
                feasible_space = V1alpha3FeasibleSpace(list=hp_parameters[parameter]['list'])
                parameters.append(V1alpha3ParameterSpec(
                    feasible_space=feasible_space,
                    name=parameter,
                    parameter_type=hp_parameters[parameter]['type']
                ))


        # 实验模板
        go_template = V1alpha3GoTemplate(
            raw_template=item.trial_spec
        )

        trial_template = V1alpha3TrialTemplate(go_template=go_template)
        labels = {
            "run-rtx":g.user.username,
            "hp-name":item.name,
            # "hp-describe": item.describe
        }
        # Experiment 跑实例测试
        experiment = V1alpha3Experiment(
            api_version= conf.get('CRD_INFO')['experiment']['group']+"/"+ conf.get('CRD_INFO')['experiment']['version'] ,#"kubeflow.org/v1alpha3",
            kind="Experiment",
            metadata=V1ObjectMeta(name=item.name+"-"+uuid.uuid4().hex[:4], namespace=conf.get('KATIB_NAMESPACE'),labels=labels),

            spec=V1alpha3ExperimentSpec(
                algorithm=algorithm,
                max_failed_trial_count=item.max_failed_trial_count,
                max_trial_count=item.max_trial_count,
                metrics_collector_spec=metrics_collector_spec,
                objective=objective,
                parallel_trial_count=item.parallel_trial_count,
                parameters=parameters,
                trial_template=trial_template
            )
        )
        item.experiment = json.dumps(experiment.to_dict(),indent=4,ensure_ascii=False)

    @expose('/create_experiment/<id>',methods=['GET'])
    # @pysnooper.snoop(watch_explode=('hp',))
    def create_experiment(self,id):
        hp = db.session.query(Hyperparameter_Tuning).filter(Hyperparameter_Tuning.id == int(id)).first()
        if hp:
            from myapp.utils.py.py_k8s import K8s
            k8s_client = K8s(hp.project.cluster.get('KUBECONFIG',''))
            namespace = conf.get('KATIB_NAMESPACE')
            crd_info =conf.get('CRD_INFO')['experiment']
            print(hp.experiment)
            k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,body=hp.experiment)
            flash('部署完成','success')

            # kclient = kc.KatibClient()
            # kclient.create_experiment(hp, namespace=conf.get('KATIB_NAMESPACE'))

        self.update_redirect()
        return redirect(self.get_redirect())



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
        if item.job_type=='TFJob':
            item.trial_spec=core.merge_tfjob_experiment_template(
                worker_num=item.tf_worker_num,
                node_selector=item.get_node_selector(),
                volume_mount=item.volume_mount,
                image=item.tf_worker_image,
                image_secrets = image_secrets,
                hostAliases=conf.get('HOSTALIASES',''),
                workingDir=item.working_dir,
                image_pull_policy=item.image_pull_policy,
                resource_memory=item.resource_memory,
                resource_cpu=item.resource_cpu,
                command=item.tf_worker_command
            )
            item.job_json={
                "tf_worker_num":item.tf_worker_num,
                "tf_worker_image": item.tf_worker_image,
                "tf_worker_command": item.tf_worker_command,
            }
        if item.job_type == 'Job':
            item.trial_spec=core.merge_job_experiment_template(
                node_selector=item.get_node_selector(),
                volume_mount=item.volume_mount,
                image=item.job_worker_image,
                image_secrets=image_secrets,
                hostAliases=conf.get('HOSTALIASES',''),
                workingDir=item.working_dir,
                image_pull_policy=item.image_pull_policy,
                resource_memory=item.resource_memory,
                resource_cpu=item.resource_cpu,
                command=item.job_worker_command
            )

            item.job_json = {
                "job_worker_image": item.job_worker_image,
                "job_worker_command": item.job_worker_command,
            }
        if item.job_type == 'PyTorchJob':
            item.trial_spec=core.merge_pytorchjob_experiment_template(
                worker_num=item.pytorch_worker_num,
                node_selector=item.get_node_selector(),
                volume_mount=item.volume_mount,
                image=item.pytorch_worker_image,
                image_secrets=image_secrets,
                hostAliases=conf.get('HOSTALIASES', ''),
                workingDir=item.working_dir,
                image_pull_policy=item.image_pull_policy,
                resource_memory=item.resource_memory,
                resource_cpu=item.resource_cpu,
                master_command=item.pytorch_master_command,
                worker_command=item.pytorch_worker_command
              )

            item.job_json = {
                "pytorch_worker_num":item.pytorch_worker_num,
                "pytorch_worker_image": item.pytorch_worker_image,
                "pytorch_master_command": item.pytorch_master_command,
                "pytorch_worker_command": item.pytorch_worker_command,
            }
        item.job_json = json.dumps(item.job_json,indent=4,ensure_ascii=False)


    # 检验参数是否有效
    # @pysnooper.snoop()
    def validate_parameters(self,parameters,algorithm):
        try:
            parameters = json.loads(parameters)
            for parameter_name in parameters:
                parameter = parameters[parameter_name]
                if parameter['type'] == 'int' and 'min' in parameter and 'max' in parameter:
                    parameter['min'] = int(parameter['min'])
                    parameter['max'] = int(parameter['max'])
                    if not parameter['max']>parameter['min']:
                        raise Exception('min must lower than max')
                    continue
                if parameter['type'] == 'double' and 'min' in parameter and 'max' in parameter:
                    parameter['min'] = float(parameter['min'])
                    parameter['max'] = float(parameter['max'])
                    if not parameter['max']>parameter['min']:
                        raise Exception('min must lower than max')
                    if algorithm=='grid':
                        parameter['step'] = float(parameter['step'])
                    continue
                if parameter['type']=='categorical' and 'list' in parameter and type(parameter['list'])==list:
                    continue

                raise MyappException('parameters type must in [int,double,categorical], and min\max\step\list should exist, and min must lower than max ')

            return json.dumps(parameters,indent=4,ensure_ascii=False)

        except Exception as e:
            print(e)
            raise MyappException('parameters not valid:'+str(e))


    # @pysnooper.snoop()
    def pre_add(self, item):
        if item.job_type is None:
            raise MyappException("Job type is mandatory")

        core.validate_json(item.parameters)
        item.parameters = self.validate_parameters(item.parameters,item.algorithm_name)

        item.resource_memory=core.check_resource_memory(item.resource_memory,self.src_item_json.get('resource_memory',None) if self.src_item_json else None)
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu,self.src_item_json.get('resource_cpu',None) if self.src_item_json else None)
        self.merge_trial_spec(item)
        self.make_experiment(item)


    def pre_update(self, item):
        self.pre_add(item)

    pre_add_get=set_column
    pre_update_get=set_column


    @action(
        "copy", __("Copy Hyperparameter Experiment"), confirmation=__('Copy Hyperparameter Experiment'), icon="fa-copy",multiple=True, single=False
    )
    def copy(self, hps):
        if not isinstance(hps, list):
            hps = [hps]
        for hp in hps:
            new_hp = hp.clone()
            new_hp.name = new_hp.name+"-copy"
            new_hp.describe = new_hp.describe + "-copy"
            new_hp.created_on = datetime.datetime.now()
            new_hp.changed_on = datetime.datetime.now()
            db.session.add(new_hp)
            db.session.commit()

        return redirect(request.referrer)

class Hyperparameter_Tuning_ModelView(Hyperparameter_Tuning_ModelView_Base,MyappModelView):
    datamodel = SQLAInterface(Hyperparameter_Tuning)
    conv = GeneralModelConverter(datamodel)


# 添加视图和菜单
appbuilder.add_view(Hyperparameter_Tuning_ModelView,"katib超参搜索",icon = 'fa-shopping-basket',category = '超参搜索',category_icon = 'fa-glass')


# 添加api
class Hyperparameter_Tuning_ModelView_Api(Hyperparameter_Tuning_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Hyperparameter_Tuning)
    conv = GeneralModelConverter(datamodel)
    route_base = '/hyperparameter_tuning_modelview/api'
    list_columns = ['created_by','changed_by','created_on','changed_on','job_type','name','namespace','describe',
                    'parallel_trial_count','max_trial_count','max_failed_trial_count','objective_type',
                    'objective_goal','objective_metric_name','objective_additional_metric_names','algorithm_name',
                    'algorithm_setting','parameters','job_json','trial_spec','working_dir','node_selector',
                    'image_pull_policy','resource_memory','resource_cpu','experiment','alert_status']
    add_columns = ['job_type','name','namespace','describe',
                    'parallel_trial_count','max_trial_count','max_failed_trial_count','objective_type',
                    'objective_goal','objective_metric_name','objective_additional_metric_names','algorithm_name',
                    'algorithm_setting','parameters','job_json','working_dir','node_selector','image_pull_policy',
                   'resource_memory','resource_cpu']
    edit_columns = add_columns
appbuilder.add_api(Hyperparameter_Tuning_ModelView_Api)



# list正在运行的Experiments
from myapp.views.view_workflow import Crd_ModelView_Base
from myapp.models.model_katib import Experiments
class Experiments_ModelView(Crd_ModelView_Base,MyappModelView,DeleteMixin):
    label_title='超参调度'
    datamodel = SQLAInterface(Experiments)
    list_columns = ['url','namespace_url','create_time','status','username']
    crd_name = 'experiment'

appbuilder.add_view(Experiments_ModelView,"katib超参调度",icon = 'fa-tasks',category = '超参搜索')



# 添加api
class Experiments_ModelView_Api(Crd_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Experiments)
    route_base = '/experiments_modelview/api'
    list_columns = ['url', 'namespace_url', 'create_time', 'status', 'username']
    crd_name = 'experiment'

appbuilder.add_api(Experiments_ModelView_Api)



