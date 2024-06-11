from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from myapp.models.model_train_model import Training_Model
from myapp.models.model_serving import InferenceService
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder, db
import uuid
from myapp.views.view_team import Project_Join_Filter

from wtforms.validators import DataRequired, Length, Regexp
from wtforms import SelectField, StringField
from flask_appbuilder.fieldwidgets import Select2Widget
from myapp.forms import MyBS3TextFieldWidget
from flask import (
    flash,
    g,
    redirect, request
)
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
from .baseApi import (
    MyappModelRestApi
)

from flask_appbuilder import expose
import datetime, json

conf = app.config


class Training_Model_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query
        return query.filter(self.model.created_by_fk == g.user.id)


class Training_Model_ModelView_Base():
    datamodel = SQLAInterface(Training_Model)
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('changed_on', 'desc')
    order_columns = ['id']
    list_columns = ['project_url', 'name', 'version', 'model_metric', 'framework', 'api_type', 'pipeline_url',
                    'creator', 'modified', 'deploy']
    search_columns = ['created_by', 'project', 'name', 'version', 'framework', 'api_type', 'pipeline_id', 'run_id',
                      'path']
    add_columns = ['project', 'name', 'version', 'describe', 'path', 'framework', 'run_id', 'run_time', 'metrics',
                   'md5', 'api_type', 'pipeline_id']
    edit_columns = add_columns
    show_columns = add_columns
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields
    cols_width = {
        "name": {"type": "ellip2", "width": 250},
        "project_url": {"type": "ellip2", "width": 200},
        "pipeline_url": {"type": "ellip2", "width": 300},
        "version": {"type": "ellip2", "width": 200},
        "modified": {"type": "ellip2", "width": 150},
        "deploy": {"type": "ellip2", "width": 100},
        "model_metric": {"type": "ellip2", "width": 300},
    }
    spec_label_columns = {
        "path": _("模型文件"),
        "framework": _("训练框架"),
        "api_type": _("推理框架"),
        "deploy": _("发布")
    }

    label_title = _('模型')
    base_filters = [["id", Training_Model_Filter, lambda: []]]

    path_describe = _('''
serving：自定义镜像的推理服务，模型地址随意<br>
tfserving：仅支持添加了服务签名的saved_model目录地址，例如：/mnt/xx/../saved_model/<br>
torch-server：torch-model-archiver编译后的mar模型文件，需保存模型结构和模型参数，例如：/mnt/xx/../xx.mar或torch script保存的模型<br>
onnxruntime：onnx模型文件的地址，例如：/mnt/xx/../xx.onnx<br>
triton-server：框架:地址。onnx:模型文件地址model.onnx，pytorch:torchscript模型文件地址model.pt，tf:模型目录地址saved_model，tensorrt:模型文件地址model.plan
'''.strip())

    service_type_choices = [x.replace('_', '-') for x in ['serving','tfserving', 'torch-server', 'onnxruntime', 'triton-server','aihub']]

    add_form_extra_fields = {
        "path": StringField(
            _('模型文件地址'),
            default='/mnt/admin/xx/saved_model/',
            description=path_describe,
            validators=[DataRequired()]
        ),
        "describe": StringField(
            _("描述"),
            description= _('模型描述'),
            validators=[DataRequired()]
        ),
        "pipeline_id": StringField(
            _('任务流id'),
            description= _('任务流的id，0表示非任务流产生模型'),
            default='0'
        ),
        "version": StringField(
            _('版本'),
            widget=MyBS3TextFieldWidget(),
            description= _('模型版本'),
            default=datetime.datetime.now().strftime('v%Y.%m.%d.1'),
            validators=[DataRequired()]
        ),
        "run_id": StringField(
            _('run id'),
            widget=MyBS3TextFieldWidget(),
            description= _('pipeline 训练的run id'),
            default='random_run_id_' + uuid.uuid4().hex[:32]
        ),
        "run_time": StringField(
            _('运行时间'),
            widget=MyBS3TextFieldWidget(),
            description= _('pipeline 训练的 运行时间'),
            default=datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
        ),
        "name": StringField(
            _("模型名"),
            widget=MyBS3TextFieldWidget(),
            description= _('模型名(a-z0-9-字符组成，最长54个字符)'),
            validators=[DataRequired(), Regexp("^[a-z0-9\-]*$"), Length(1, 54)]
        ),
        "framework": SelectField(
            _('算法框架'),
            description= _("选项xgb、tf、pytorch、onnx、tensorrt等"),
            widget=Select2Widget(),
            choices=[['sklearn','sklearn'],['xgb', 'xgb'], ['tf', 'tf'], ['pytorch', 'pytorch'], ['onnx', 'onnx'], ['tensorrt', 'tensorrt'],['aihub', 'aihub']],
            validators=[DataRequired()]
        ),
        'api_type': SelectField(
            _("部署类型"),
            description= _("推理框架类型"),
            choices=[[x, x] for x in service_type_choices],
            validators=[DataRequired()]
        )
    }
    edit_form_extra_fields = add_form_extra_fields

    # edit_form_extra_fields['path']=FileField(
    #         __('模型压缩文件'),
    #         description=_(path_describe),
    #         validators=[
    #             FileAllowed(["zip",'tar.gz'],_("zip/tar.gz Files Only!")),
    #         ]
    #     )

    # @pysnooper.snoop(watch_explode=('item'))
    def pre_add(self, item):
        if not item.run_id:
            item.run_id = 'random_run_id_' + uuid.uuid4().hex[:32]
        if not item.pipeline_id:
            item.pipeline_id = 0

    def pre_update(self, item):
        if not item.path:
            item.path = self.src_item_json['path']
        self.pre_add(item)

    import pysnooper
    @expose("/download/<model_id>", methods=["GET", 'POST'])
    # @pysnooper.snoop()
    def download_model(self, model_id):
        train_model = db.session.query(Training_Model).filter_by(id=model_id).first()
        if train_model.download_url:
            return redirect(train_model.download_url)
        if train_model.path:
            if 'http://' in train_model.path or 'https://' in train_model.path:
                return redirect(train_model.path)
            if '/mnt' in train_model.path:
                download_url = request.host_url + 'static/' + train_model.path.strip('/')
                return redirect(download_url)
        flash(__('未发现模型存储地址'),'warning')

        return redirect(conf.get('train_model'))


    @expose("/deploy/<model_id>", methods=["GET", 'POST'])
    def deploy(self, model_id):
        train_model = db.session.query(Training_Model).filter_by(id=model_id).first()
        exist_inference = db.session.query(InferenceService).filter_by(model_name=train_model.name).filter_by(model_version=train_model.version).first()
        from myapp.views.view_inferenceserving import InferenceService_ModelView_base
        inference_class = InferenceService_ModelView_base()
        inference_class.src_item_json = {}
        if not exist_inference:
            exist_inference = InferenceService()
            exist_inference.project_id = train_model.project_id
            exist_inference.project = train_model.project
            exist_inference.model_name = train_model.name
            exist_inference.label = train_model.describe
            exist_inference.model_version = train_model.version
            exist_inference.model_path = train_model.path
            exist_inference.service_type = train_model.api_type
            exist_inference.images = ''
            exist_inference.name = '%s-%s-%s' % (exist_inference.service_type, train_model.name, train_model.version.replace('v', '').replace('.', ''))
            inference_class.pre_add(exist_inference)

            db.session.add(exist_inference)
            db.session.commit()
            flash(__('新服务版本创建完成'), 'success')
        else:
            flash(__('服务版本已存在'), 'success')
        import urllib.parse

        url = conf.get('MODEL_URLS', {}).get('inferenceservice', '') + '?filter=' + urllib.parse.quote(json.dumps([{"key": "model_name", "value": exist_inference.model_name}], ensure_ascii=False))
        print(url)
        return redirect(url)


class Training_Model_ModelView(Training_Model_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Training_Model)
    route_base = '/training_model_modelview/web/api'
    add_columns = ['project', 'name', 'version', 'describe', 'path', 'framework', 'metrics','api_type']


appbuilder.add_api(Training_Model_ModelView)


class Training_Model_ModelView_Api(Training_Model_ModelView_Base, MyappModelRestApi):  # noqa
    datamodel = SQLAInterface(Training_Model)
    # base_order = ('id', 'desc')
    route_base = '/training_model_modelview/api'


appbuilder.add_api(Training_Model_ModelView_Api)
