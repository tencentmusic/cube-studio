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
import urllib
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
        "service_config": "推理配置",
        "session_num":"上下文记录个数"
    }

    list_columns = ['label', 'chat_type', 'service_type']
    cols_width = {
        "name": {"type": "ellip1", "width": 100},
        "label": {"type": "ellip2", "width": 200},

        "chat_type": {"type": "ellip1", "width": 100},
        "hello": {"type": "ellip1", "width": 200},
        "service_type": {"type": "ellip1", "width": 100}
    }
    icon='<svg t="1679629996403" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3590" width="200" height="200"><path d="M291.584 806.314667c-13.909333 0-25.6-11.690667-25.6-25.6v-145.92c0-135.68 110.336-246.016 246.016-246.016s246.016 110.336 246.016 246.016v145.92c0 13.909333-11.690667 25.6-25.6 25.6H291.584z" fill="#64EDAC" p-id="3591"></path><path d="M627.114667 626.517333c-18.773333 0-34.133333-15.36-34.133334-34.133333v-36.096c0-18.773333 15.36-34.133333 34.133334-34.133333s34.133333 15.36 34.133333 34.133333v36.096c0 18.773333-15.36 34.133333-34.133333 34.133333zM396.885333 626.517333c-18.773333 0-34.133333-15.36-34.133333-34.133333v-36.096c0-18.773333 15.36-34.133333 34.133333-34.133333s34.133333 15.36 34.133334 34.133333v36.096c0 18.773333-15.36 34.133333-34.133334 34.133333z" fill="#333C4F" p-id="3592"></path><path d="M735.829333 356.949333l44.288-77.226666c23.552 0.085333 46.506667-12.202667 59.306667-34.389334 19.029333-33.194667 8.106667-75.776-24.490667-95.232-32.512-19.370667-74.325333-8.192-93.354666 24.917334-12.714667 22.186667-11.946667 48.64-0.341334 69.546666l-42.752 74.496a355.1488 355.1488 0 0 0-166.4-41.301333 355.072 355.072 0 0 0-159.744 37.888l-40.704-70.997333a70.8608 70.8608 0 0 0-0.341333-69.546667c-19.029333-33.194667-60.842667-44.373333-93.354667-24.917333-32.512 19.370667-43.52 62.037333-24.490666 95.232a68.027733 68.027733 0 0 0 59.306666 34.389333l41.557334 72.533333c-84.650667 65.194667-139.264 167.509333-139.264 282.368v145.92c0 75.264 61.269333 136.533333 136.533333 136.533334h440.917333c75.264 0 136.533333-61.269333 136.533334-136.533334v-145.92c-0.085333-112.128-52.053333-212.309333-133.205334-277.76z m64.853334 423.765334c0 37.632-30.634667 68.266667-68.266667 68.266666H291.584c-37.632 0-68.266667-30.634667-68.266667-68.266666v-145.92c0-159.232 129.536-288.682667 288.682667-288.682667 159.232 0 288.682667 129.536 288.682667 288.682667v145.92z" fill="#333C4F" p-id="3593"></path><path d="M580.266667 794.88H443.733333c-18.773333 0-34.133333-15.36-34.133333-34.133333V759.466667c0-18.773333 15.36-34.133333 34.133333-34.133334h136.533334c18.773333 0 34.133333 15.36 34.133333 34.133334v1.28c0 18.773333-15.36 34.133333-34.133333 34.133333z" fill="#333C4F" p-id="3594"></path><path d="M553.642667 237.994667c-9.898667 0-19.2-4.266667-25.685334-11.690667l-26.112-29.866667-18.602666 26.88a34.2528 34.2528 0 0 1-28.074667 14.762667H395.093333c-18.858667 0-34.133333-15.274667-34.133333-34.133333s15.274667-34.133333 34.133333-34.133334h42.24l33.365334-48.298666c5.973333-8.704 15.616-14.08 26.197333-14.677334 10.581333-0.597333 20.736 3.754667 27.648 11.690667l44.629333 51.2 68.266667-0.426667h0.256a34.1248 34.1248 0 0 1 0.256 68.266667l-83.968 0.512c-0.170667-0.085333-0.256-0.085333-0.341333-0.085333z" fill="#333C4F" p-id="3595"></path></svg>'
    service_config={
        "url":"http://xx.xx:xx/xx",
        "headers":{

        },
        "data":{

        }
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
        "icon": StringField(
            label=_(datamodel.obj.lab('icon')),
            default=icon,
            description='图标',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

        "chat_type": SelectField(
            label=_(datamodel.obj.lab('field')),
            description='对话类型',
            default='文本对话',
            widget=MySelect2Widget(),
            choices=[['text','文本对话'],['image',"图片"],  ['text_speech',"文本+语音"],['multi_modal', "多模态"], ['digital_human',"数字人"]],
            validators=[]
        ),
        "service_type": SelectField(
            label=_(datamodel.obj.lab('source_type')),
            description='推理服务类型',
            widget=Select2Widget(),
            default='chatgpt',
            choices=[[x, x] for x in ["chatgpt3.5","chatgpt4", "chatglm", "微调模型",'AIGC','组合']],
            validators=[]
        ),
        "service_config": StringField(
            label=_(datamodel.obj.lab('service_config')),
            default=json.dumps(service_config,indent=4,ensure_ascii=False),
            description='服务配置',
            widget=MyBS3TextAreaFieldWidget(),
            validators=[DataRequired()]
        )
    }

    edit_form_extra_fields = add_form_extra_fields
    all_history={}
    @expose('/chat/<chat_id>', methods=['POST','GET'])
    @pysnooper.snoop()
    def chat(self,chat_id):
        request_id = request.json.get('request_id')
        search_text = request.json.get('search_text','')
        search_audio = request.json.get('search_audio',None)
        chat= db.session.query(Chat).filter_by(id=int(chat_id)).first()
        if request_id not in self.all_history:
            self.all_history[request_id]=[]
        history = self.all_history.get(request_id, [])
        if chat.session_num:
            session_num = int(chat.session_num)
            if session_num>0:
                history=history[0-session_num:]

        if chat.service_type.lower()=='chatglm':
            status,text = self.chatglm(chat=chat,search_text=search_text,history=history)
            # 追加记录的方法
            self.all_history[request_id].append(search_text)
            return jsonify({
                "status":status,
                "message":'失败' if status else "成功",
                "result":[
                        {
                            "text":text
                        }
                    ]
            })
        if chat.service_type.lower()=='chatgpt':
            status,text = self.chatgpt(chat=chat,search_text=search_text,history=history)
            # 追加记录的方法
            self.all_history[request_id].append(search_text)
            return jsonify({
                "status":status,
                "message":'失败' if status else "成功",
                "result":[
                        {
                            "text":text
                        }
                    ]
            })
        if chat.service_type.lower()=='aigc':
            status,image = self.aigc(chat=chat,search_text=search_text,history=history)
            # 追加记录的方法
            self.all_history[request_id].append(search_text)
            return jsonify({
                "status":status,
                "message":'失败' if status else "成功",
                "result":[
                        {
                            "image":image
                        }
                    ]
            })
        if chat.service_type.lower()=='组合':
            if chat.chat_type=='text_speech':
                # 发送的是文字，就返回文字
                if search_text:
                    status,message = self.chatglm(chat=chat,search_text=search_text,history=history)

                    return jsonify({
                        "status": status,
                        "message": 'chatglm失败' if status else "chatglm成功",
                        "result": [
                            {
                                "text": message
                            }
                        ]
                    })
                # 发送的是语音，就返回文字+语音
                elif search_audio:
                    status,message = self.asr(chat=chat,search_audio=search_audio,history=history)
                    if status:
                        return jsonify({
                            "status": status,
                            "message": 'asr失败' if status else "asr成功",
                            "result": [
                                {
                                    "text": message
                                }
                            ]
                        })
                    status,message = self.chatglm(chat=chat, search_text=message, history=history)
                    if status:
                        return jsonify({
                            "status": status,
                            "message": 'chatglm失败' if status else "chatglm成功",
                            "result": [
                                {
                                    "text": message
                                }
                            ]
                        })
                    status,audio = self.tts(chat=chat, search_text=message, history=history)
                    print(audio)
                    return jsonify({
                        "status":status,
                        "message":'tts失败' if status else "tts成功",
                        "result":[
                            {
                                "text":message,
                                "audio":audio
                            }
                        ]
                    })



    # @pysnooper.snoop()
    def chatgpt(self,chat,search_text,history=[]):
        try:
            url = json.loads(chat.service_config).get("chatgpt_url", '')
            headers = json.loads(chat.service_config).get("chatgpt_headers", {})
            data = {
                "msgContent": chat.pre_question + search_text,
                "msgType": "text",
                "chatType": "group",
                "userName": "zhang-san",
                "botName": "text",
                "botKey": "c1534e3f-****-4d21-****-484bc1a1307f",
                "hookUrl": "string",
                "msgId": "fCgAA1AxOagtYsQ5R",
                "chatInfoUrl": "string",
                "eventType": "add_to_chat"
            }
            data.update(json.loads(chat.service_config).get("chatgpt_data", {}))
            res = requests.post(
                url,
                headers=headers,
                json=data
            )
            if res.status_code == 200:
                mes = res.json().get("msgContent", '')
                mes = mes[:mes.index('\n')]
                return 0,mes
            else:
                return 1, f'请求{url}失败'
        except Exception as e:
            return 1,"chatgpt报错："+str(e)



    # @pysnooper.snoop()
    def chatglm(self,chat,search_text,history=[]):
        try:
            query={
                "query":chat.pre_question+search_text
            }
            url = json.loads(chat.service_config).get("chatglm_url",'')
            res = requests.post(url,json=query)
            if res.status_code==200:
                result = res.json().get("result", [])
                if result:
                    result = result[0]['markdown']
                    return 0,result
                return 1,res.json().get("message", [])
            else:
                return 1, f'请求{url}失败'
        except Exception as e:
            return 1,'chatglm报错：'+str(e)


    @pysnooper.snoop()
    def tts(self, chat, search_text, history=[]):
        try:
            url = json.loads(chat.service_config).get("tts_url", '')
            headers = json.loads(chat.service_config).get("tts_headers", {})
            data = {
                "text": search_text
            }
            data.update(json.loads(chat.service_config).get("tts_data", {}))
            res = requests.post(url,
                                headers=headers,
                                json=data
                                )
            if res.status_code==200:
                result = res.json().get("result", [])
                print(result)
                if result:
                    audio = result[0]['audio']
                    if 'http:' not in audio and 'https://' not in audio:
                        audio = urllib.parse.urljoin(url, audio)
                        print(audio)
                    return 0,audio
                return 1,result.get('message')
            else:
                return 1, f'请求{url}失败'
        except Exception as e:
            return 1,'tts报错：'+str(e)

    @pysnooper.snoop()
    def asr(self, chat, search_audio, history=[]):
        # return 0,'北京在哪里？'
        try:
            url = json.loads(chat.service_config).get("asr_url", '')
            headers = json.loads(chat.service_config).get("asr_headers", {})
            data = {
                "voice_file_path": search_audio
            }
            data.update(json.loads(chat.service_config).get("asr_data", {}))
            res = requests.post(url,
                                headers=headers,
                                json=data
                                )
            if res.status_code==200:
                result = res.json().get("result", [])
                if result:
                    result = result[0]['text']
                    return 0,result
                return 1,res.json().get("message", [])
            else:
                return 1, f'请求{url}失败'
        except Exception as e:
            return 1,"asr报错：" + str(e)

    @pysnooper.snoop()
    def aigc(self,chat,search_text,history=[]):
        try:
            url = json.loads(chat.service_config).get("aigc_url",'')
            headers = json.loads(chat.service_config).get("aigc_headers",{})
            data = {
                "text": chat.pre_question + search_text
            }
            data.update(json.loads(chat.service_config).get("aigc_data",{}))
            res = requests.post(
                url,
                headers=headers,
                json=data
            )
            if res.status_code==200:
                result = res.json().get("result", [])
                if result:
                    image = result[0]['image']
                    if 'http:' not in image and 'https://' not in image:
                        image = urllib.parse.urljoin(url, image)
                        print(image)
                    return 0,image
                return 1,result.get("message")
            else:
                return 1, f'请求{url}失败'
        except Exception as e:
            return 1,'aigc报错：'+str(e)

appbuilder.add_api(Chat_View)



