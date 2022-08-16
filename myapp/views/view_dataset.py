import traceback

from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.forms import GeneralModelConverter
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for
import uuid
from flask import Blueprint, current_app, jsonify, make_response, request
from flask_appbuilder.actions import action
import re,os
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError
from myapp import app, appbuilder,db,event_logger
from myapp.utils import core
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField,MySelect2ManyWidget
from wtforms.ext.sqlalchemy.fields import QuerySelectField

from .baseApi import (
    MyappModelRestApi,
    json_response
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
from marshmallow import ValidationError
from sqlalchemy.exc import IntegrityError
from myapp.models.model_dataset import Dataset
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
from myapp.security import MyUser
conf = app.config
logging = app.logger





class Dataset_ModelView_base():
    label_title='数据集'
    datamodel = SQLAInterface(Dataset)
    base_permissions = ['can_add','can_show','can_edit','can_list','can_delete']

    base_order = ("id", "desc")
    order_columns=['id']

    add_columns = ['name','label','describe','source_type','source','industry','field','usage','research','storage_class','file_type','years','url','download_url','path','storage_size','entries_num','duration','price','status']
    show_columns = ['id','name','label','describe','source_type','source','industry','field','usage','research','storage_class','file_type','status','years','url','path','download_url','storage_size','entries_num','duration','price']
    search_columns=['name','label','describe','source_type','source','industry','field','usage','research','storage_class','file_type','status','years','url','path','download_url']
    spec_label_columns = {
        "source_type":"来源类型",
        "source":"数据来源",
        "usage":"数据用途",
        "research":"研究方向",
        "storage_class":"存储类型",
        "file_type":"文件类型",
        "years":"数据年份",
        "url":"相关网址",
        "url_html": "相关网址",
        "download_url":"下载地址",
        "download_url_html": "下载地址",
        "path":"本地路径",
        "entries_num":"条目数量",
        "duration":"文件时长",
        "price": "价格"
    }

    edit_columns = add_columns
    list_columns = ['name','label','describe','source_type','source','status','industry','field','url_html','download_url_html','usage','research','storage_class','file_type','years','path','storage_size','entries_num','duration','price','owner']
    cols_width = {
        "name": {"type": "ellip1", "width": 250},
        "label": {"type": "ellip1", "width": 300},
        "describe":{"type": "ellip1", "width": 300},
        "field":{"type": "ellip1", "width": 100},
        "source_type":{"type": "ellip1", "width": 100},
        "source": {"type": "ellip1", "width": 100},
        "industry": {"type": "ellip1", "width": 100},
        "url_html": {"type": "ellip1", "width": 200},
        "download_url_html": {"type": "ellip1", "width": 200},
        "path":{"type": "ellip1", "width": 200},
        "storage_class": {"type": "ellip1", "width": 100},
        "storage_size":{"type": "ellip1", "width": 100},
        "file_type":{"type": "ellip1", "width": 100},
        "owner": {"type": "ellip1", "width": 200},
        "status": {"type": "ellip1", "width": 100},
        "entries_num": {"type": "ellip1", "width": 100},
        "duration": {"type": "ellip1", "width": 100},
        "price": {"type": "ellip1", "width": 100},
        "years": {"type": "ellip2", "width": 100},
        "usage": {"type": "ellip1", "width": 100},
        "research": {"type": "ellip1", "width": 100},
    }

    add_form_extra_fields = {
        "name": StringField(
            label=_(datamodel.obj.lab('name')),
            description='数据集英文名',
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "label": StringField(
            label=_(datamodel.obj.lab('label')),
            default='',
            description='数据集中文名',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "describe": StringField(
            label=_(datamodel.obj.lab('describe')),
            default='',
            description='数据集描述',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "industry": SelectField(
            label=_(datamodel.obj.lab('industry')),
            description='行业分类',
            widget=MySelect2Widget(can_input=True),
            default='',
            choices=[[x,x] for x in ['农业','生物学','气候+天气','复杂网络','计算机网络','网络安全','数据挑战','地球科学','经济学','教育','能源','娱乐','金融','GIS','政府','医疗','图像处理','机器学习','博物馆','自然语言','神经科学','物理','前列腺癌','心理学+认知','公共领域','搜索引擎','社交网络','社会科学','软件','运动','时间序列','交通','电子竞技']],
            validators=[DataRequired()]
        ),
        "field":SelectField(
            label=_(datamodel.obj.lab('field')),
            description='领域',
            widget=MySelect2Widget(can_input=True),
            choices=[[x,x] for x in ['视觉',"音频","自然语言","风控","搜索",'推荐']],
            validators=[]
        ),
        "source_type": SelectField(
            label=_(datamodel.obj.lab('source_type')),
            description='来源分类',
            widget=Select2Widget(),
            default='开源',
            choices=[[x,x] for x in ["开源", "自产","购买"]],
            validators=[]
        ),
        "source": SelectField(
            label=_(datamodel.obj.lab('source')),
            description='数据来源',
            widget=MySelect2Widget(can_input=True),
            choices=[[x, x] for x in ['github',"kaggle", "天池",'UCI','AWS 公开数据集','Google 公开数据集',  "采购公司1", "标注团队1", "政府网站1"]],
            validators=[]
        ),
        "file_type": SelectField(
            label=_(datamodel.obj.lab('file_type')),
            description='文件类型',
            widget=MySelect2Widget(can_input=True),
            choices=[[x, x] for x in ["png", "jpg",'txt','csv','wav','mp3','mp4','nv4']],
        ),
        "storage_class": SelectField(
            label=_(datamodel.obj.lab('storage_class')),
            description='存储类型',
            widget=MySelect2Widget(can_input=True),
            choices=[[x, x] for x in ["压缩", "未压缩"]],
        ),
        "storage_size": StringField(
            label=_(datamodel.obj.lab('storage_size')),
            description='存储大小',
            widget=BS3TextFieldWidget(),
        ),
        "owner": StringField(
            label=_(datamodel.obj.lab('owner')),
            default='',
            description='责任人,逗号分隔的多个用户',
            widget=BS3TextFieldWidget(),
        ),
        "status": SelectField(
            label=_(datamodel.obj.lab('status')),
            description='数据集状态',
            widget=MySelect2Widget(can_input=True),
            choices=[[x, x] for x in ["损坏", "正常",'未购买','已购买','未标注','已标注','未校验','已校验']],
        ),
    }
    edit_form_extra_fields = add_form_extra_fields


    import_data=True

    def post_list(self,items):
        flash(Markup('可批量删除不使用的数据集,可批量上传自产数据集'),category='info')
        return items

class Dataset_ModelView_Api(Dataset_ModelView_base,MyappModelRestApi):
    datamodel = SQLAInterface(Dataset)
    route_base = '/dataset_modelview/api'

appbuilder.add_api(Dataset_ModelView_Api)

