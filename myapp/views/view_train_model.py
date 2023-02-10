from flask_appbuilder.models.sqla.interface import SQLAInterface
from myapp.models.model_train_model import Training_Model
from myapp.models.model_serving import InferenceService
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder,db
import uuid
from myapp.views.view_team import Project_Join_Filter

from wtforms.validators import DataRequired, Length, Regexp
from wtforms import SelectField, StringField
from flask_appbuilder.fieldwidgets import Select2Widget
from myapp.forms import MyBS3TextFieldWidget
from flask import (
    flash,
    g,
    redirect
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
    list_columns = ['project_url','name','version','model_metric','framework','api_type','pipeline_url','creator','modified','deploy']
    search_columns = ['created_by','project','name','version','framework','api_type','pipeline_id','run_id','path']

    add_columns = ['project','name','version','describe','path','framework','run_id','run_time','metrics','md5','api_type','pipeline_id']
    edit_columns = add_columns
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields
    cols_width={
        "name":{"type": "ellip2", "width": 250},
        "project_url": {"type": "ellip2", "width": 200},
        "pipeline_url":{"type": "ellip2", "width": 300},
        "version": {"type": "ellip2", "width": 200},
        "modified": {"type": "ellip2", "width": 150},
        "deploy": {"type": "ellip2", "width": 100},
        "model_metric": {"type": "ellip2", "width": 300},
    }
    spec_label_columns = {
        "path": "模型文件",
        "framework":"算法框架",
        "api_type":"推理框架",
        "pipeline_id":"任务流id",
        "deploy": "发布",
        "model_metric":"指标"
    }

    label_title = '模型'
    base_filters = [["id", Training_Model_Filter, lambda: []]]


    path_describe= r'''
            tfserving：仅支持tf save_model方式的模型目录, /mnt/xx/../saved_model/<br>
            torch-server：torch-model-archiver编译后的mar模型文件地址, /mnt/xx/../xx.mar或torch script保存的模型<br>
            onnxruntime：onnx模型文件的地址, /mnt/xx/../xx.onnx<br>
            tensorrt:模型文件地址, /mnt/xx/../xx.plan<br>
            '''


    service_type_choices= [x.replace('_','-') for x in ['tfserving','torch-server','onnxruntime','triton-server']]

    add_form_extra_fields={
        "path": StringField(
            _('模型文件地址'),
            default='/mnt/admin/xx/saved_model/',
            description=_(path_describe),
            validators=[DataRequired()]
        ),
        "describe": StringField(
            _(datamodel.obj.lab('describe')),
            description=_('模型描述'),
            validators=[DataRequired()]
        ),
        "pipeline_id": StringField(
            _(datamodel.obj.lab('pipeline_id')),
            description=_('任务流的id，0表示非任务流产生模型'),
            default='0'
        ),
        "version": StringField(
            _('版本'),
            widget=MyBS3TextFieldWidget(),
            description='模型版本',
            default=datetime.datetime.now().strftime('v%Y.%m.%d.1'),
            validators=[DataRequired()]
        ),
        "run_id":StringField(
            _(datamodel.obj.lab('run_id')),
            widget=MyBS3TextFieldWidget(),
            description='pipeline 训练的run id',
            default='random_run_id_'+uuid.uuid4().hex[:32]
        ),
        "run_time": StringField(
            _(datamodel.obj.lab('run_time')),
            widget=MyBS3TextFieldWidget(),
            description='pipeline 训练的 运行时间',
            default=datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
        ),
        "name":StringField(
            _("模型名"),
            widget=MyBS3TextFieldWidget(),
            description='模型名(a-z0-9-字符组成，最长54个字符)',
            validators = [DataRequired(),Regexp("^[a-z0-9\-]*$"),Length(1,54)]
        ),
        "framework": SelectField(
            _('算法框架'),
            description="选项xgb、tf、pytorch、onnx、tensorrt等",
            widget=Select2Widget(),
            choices=[['xgb', 'xgb'],['tf', 'tf'], ['pytorch', 'pytorch'],['onnx','onnx'],['tensorrt','tensorrt']],
            validators=[DataRequired()]
        ),
        'api_type': SelectField(
            _("部署类型"),
            description="推理框架类型",
            choices=[[x, x] for x in service_type_choices],
            validators=[DataRequired()]
        )
    }
    edit_form_extra_fields=add_form_extra_fields
    # edit_form_extra_fields['path']=FileField(
    #         _('模型压缩文件'),
    #         description=_(path_describe),
    #         validators=[
    #             FileAllowed(["zip",'tar.gz'],_("zip/tar.gz Files Only!")),
    #         ]
    #     )


    # @pysnooper.snoop(watch_explode=('item'))
    def pre_add(self,item):
        if not item.run_id:
            item.run_id='random_run_id_'+uuid.uuid4().hex[:32]

    def pre_update(self,item):
        if not item.path:
            item.path=self.src_item_json['path']
        self.pre_add(item)

    @expose("/deploy/<model_id>", methods=["GET",'POST'])
    def deploy(self,model_id):
        train_model = db.session.query(Training_Model).filter_by(id=model_id).first()
        exist_inference = db.session.query(InferenceService).filter_by(model_name=train_model.name).filter_by(model_version=train_model.version).first()
        from myapp.views.view_inferenceserving import InferenceService_ModelView_base
        inference_class = InferenceService_ModelView_base()
        inference_class.src_item_json={}
        if not exist_inference:
            exist_inference = InferenceService()
            exist_inference.project_id=train_model.project_id
            exist_inference.project = train_model.project
            exist_inference.model_name=train_model.name
            exist_inference.label = train_model.describe
            exist_inference.model_version=train_model.version
            exist_inference.model_path=train_model.path
            exist_inference.service_type=train_model.api_type
            exist_inference.images=''
            exist_inference.name='%s-%s-%s'%(exist_inference.service_type,train_model.name,train_model.version.replace('v','').replace('.',''))
            inference_class.pre_add(exist_inference)

            db.session.add(exist_inference)
            db.session.commit()
            flash('新服务版本创建完成','success')
        else:
            flash('服务版本已存在', 'success')
        import urllib.parse

        url = conf.get('MODEL_URLS',{}).get('inferenceservice','')+'?filter='+urllib.parse.quote(json.dumps([{"key":"model_name","value":exist_inference.model_name}],ensure_ascii=False))
        print(url)
        return redirect(url)


class Training_Model_ModelView(Training_Model_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Training_Model)

appbuilder.add_view_no_menu(Training_Model_ModelView)



class Training_Model_ModelView_Api(Training_Model_ModelView_Base,MyappModelRestApi):  # noqa
    datamodel = SQLAInterface(Training_Model)
    # base_order = ('id', 'desc')
    route_base = '/training_model_modelview/api'


appbuilder.add_api(Training_Model_ModelView_Api)


