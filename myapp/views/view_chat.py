from myapp.models.model_chat import Chat
import requests

from flask_appbuilder import action
from flask_appbuilder.models.sqla.interface import SQLAInterface
from wtforms.validators import DataRequired,Regexp
from myapp import app, appbuilder
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget
from flask import jsonify,Markup,make_response
from .baseApi import MyappModelRestApi
from flask import g,request,redirect
import json,os,sys
from werkzeug.utils import secure_filename
import pysnooper
from sqlalchemy import or_
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder,db
from flask_appbuilder import expose
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)

conf = app.config
logging = app.logger

class Chat_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        return query.filter(
            or_(
                self.model.owner.contains(g.user.username),
                self.model.owner.contains('*')
            )
        )


class Chat_View(MyappModelRestApi):
    datamodel = SQLAInterface(Chat)
    route_base = '/chat_modelview/api'
    label_title = '聊天窗配置'
    base_order = ("id", "desc")
    order_columns = ['id']
    base_filters = [["id", Chat_Filter, lambda: []]]  # 设置权限过滤器

    spec_label_columns = {
        "icon": "图标",
        "name": "英文名",
        "label": "中文名",
        "doc": "帮助文档地址",
        "chat_type": "聊天窗类型",
        "hello": "欢迎语",
        "pre_question": "先验知识",
        "service_type": "推理服务类型",
        "service_url": "推理服务地址",
        "session_num":"上下文记录个数"
    }

    list_columns = ['label', 'chat_type', 'service_type']
    cols_width = {
        "name": {"type": "ellip1", "width": 100},
        "label": {"type": "ellip2", "width": 200},

        "chat_type": {"type": "ellip1", "width": 100},
        "hello": {"type": "ellip1", "width": 200},
        "service_type": {"type": "ellip1", "width": 100},
        "service_url": {"type": "ellip2", "width": 200}
    }

    add_form_extra_fields = {
        "name": StringField(
            label=_(datamodel.obj.lab('name')),
            description='英文名，小写',
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9_]*[a-z0-9]$"), ]
        ),
        "label": StringField(
            label=_(datamodel.obj.lab('label')),
            default='',
            description='中文名',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

        "chat_type": SelectField(
            label=_(datamodel.obj.lab('field')),
            description='对话类型',
            default='文本对话',
            widget=MySelect2Widget(),
            choices=[['text','文本对话'], ['speech',"语音对话"],['image',"图片"],  ['text_speech',"文本+语音"],['multi_modal', "多模态"], ['digital_human',"数字人"]],
            validators=[]
        ),
        "service_type": SelectField(
            label=_(datamodel.obj.lab('source_type')),
            description='推理服务类型',
            widget=Select2Widget(),
            default='chatgpt',
            choices=[[x, x] for x in ["chatgpt3.5","chatgpt4", "chatglm", "微调模型",'AIGC']],
            validators=[]
        )
    }
    edit_form_extra_fields = add_form_extra_fields
    all_history={}
    @expose('/chat/<chat_id>', methods=['POST','GET'])
    def chat(self,chat_id):
        request_id = request.json.get('request_id')
        search_data = request.json.get('search_data')
        chat= db.session.query(Chat).filter_by(id=int(chat_id)).first()
        if request_id not in self.all_history:
            self.all_history[request_id]=[]
        history = self.all_history.get(request_id, [])
        if chat.session_num:
            session_num = int(chat.session_num)
            if session_num>0:
                history=history[0-session_num:]

        if chat.service_type=='chatglm':
            result = self.chatglm(chat=chat,search_data=search_data,history=history)
            # 追加记录的方法
            self.all_history[request_id].append(search_data)
            return jsonify({
                "status":0,
                "message":"成功",
                "result":result
            })
        if chat.service_type=='chatgpt':
            result = self.chatgpt(chat=chat,search_data=search_data,history=history)
            # 追加记录的方法
            self.all_history[request_id].append(search_data)
            return jsonify({
                "status":0,
                "message":"成功",
                "result":result
            })

    # http://10.101.140.141/aihub/chatglm
    def chatglm(self,chat,search_data,history=[]):
        try:
            query={
                "query":chat.pre_question+search_data
            }
            res = requests.post(chat.service_url,json=query)
            result = res.json().get("result", [])
            if result:
                result = result[0].get('markdown','')
            else:
                result = "聊天报错："+res.json().get('message','')

            return result
        except Exception as e:
            return "聊天报错："+str(e)


    @pysnooper.snoop()
    def chatgpt(self,chat,search_data,history=[]):
        try:
            res = requests.post(chat.service_url,
                                headers={
                                    "x-token": "4fc6b7d6-8c07-41db-8161-7c6365ce5214"
                                },
                                json={
                                    "msgContent": chat.pre_question+search_data,
                                    "msgType": "text",
                                    "chatType": "group",
                                    "chatId": "wrkSFfCgAA1AxOagtYsQ5R******-iw",
                                    "userName": "zhang-san",
                                    "botName": "text",
                                    "botKey": "c1534e3f-****-4d21-****-484bc1a1307f",
                                    "hookUrl": "string",
                                    "msgId": "fCgAA1AxOagtYsQ5R",
                                    "chatInfoUrl": "string",
                                    "eventType": "add_to_chat"
                                }
                                )
            mes = res.json().get("msgContent", '')
            mes = mes[:mes.index('\n')]
            return mes
        except Exception as e:
            return "聊天报错："+str(e)


    @pysnooper.snoop()
    def aigc(self,chat,search_data,history=[]):
        try:
            res = requests.post(chat.service_url,
                                headers={
                                    "x-token": "4fc6b7d6-8c07-41db-8161-7c6365ce5214"
                                },
                                json={
                                    "msgContent": chat.pre_question+search_data,
                                    "msgType": "text",
                                    "chatType": "group",
                                    "chatId": "wrkSFfCgAA1AxOagtYsQ5R******-iw",
                                    "userName": "zhang-san",
                                    "botName": "text",
                                    "botKey": "c1534e3f-****-4d21-****-484bc1a1307f",
                                    "hookUrl": "string",
                                    "msgId": "fCgAA1AxOagtYsQ5R",
                                    "chatInfoUrl": "string",
                                    "eventType": "add_to_chat"
                                }
                                )
            mes = res.json().get("msgContent", '')
            mes = mes[:mes.index('\n')]
            return mes
        except Exception as e:
            return "聊天报错："+str(e)

appbuilder.add_api(Chat_View)



