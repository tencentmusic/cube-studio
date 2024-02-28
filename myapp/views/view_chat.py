import base64
import uuid
import random
import re
import shutil
import logging
from myapp.models.model_chat import Chat, ChatLog
import requests
import time
from myapp.forms import MySelect2Widget, MyBS3TextFieldWidget
import multiprocessing
from flask import Flask, render_template, send_file
import pandas as pd
from myapp.exceptions import MyappException
from sqlalchemy.exc import InvalidRequestError
import datetime
from flask import Response,flash,g
from flask_appbuilder import action
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from wtforms.validators import DataRequired, Regexp
from myapp import app, appbuilder
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget
from flask import jsonify, Markup, make_response, stream_with_context
from .baseApi import MyappModelRestApi
from flask import g, request, redirect
import urllib
import json, os, sys
import emoji,re
from werkzeug.utils import secure_filename
import pysnooper
from sqlalchemy import or_
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder, db
from flask_appbuilder import expose
import threading
import queue
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
from myapp import cache
conf = app.config
logging.getLogger("sseclient").setLevel(logging.INFO)
max_len=2000

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


default_icon = '<svg t="1708691376697" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4274" width="50" height="50"><path d="M512 127.0272c-133.4144 0-192.4864 72.1792-192.4864 192.4864V656.384h384.9856V319.5136c-0.0128-120.3072-78.72-192.4864-192.4992-192.4864z" fill="#A67C52" p-id="4275"></path><path d="M560.128 487.9488h-96.256l-24.0512 96.2432v312.8064h144.3584V584.192z" fill="#DBB59A" p-id="4276"></path><path d="M223.2576 704.4992c-45.2352 45.2352-48.128 192.4864-48.128 192.4864H512L415.7568 608.256s-169.8944 73.6256-192.4992 96.2432z m577.4848 0C778.112 681.8688 608.256 608.256 608.256 608.256L512 896.9984h336.8576s-2.88-147.264-48.1152-192.4992z" fill="#48A0DC" p-id="4277"></path><path d="M584.1792 584.192L512 896.9984h24.064l72.1792-168.4352 120.3072-24.064-144.3712-120.3072z m-288.7296 120.3072l120.3072 24.064 72.1792 168.4352H512L439.8208 584.192l-144.3712 120.3072z" fill="#FFFFFF" p-id="4278"></path><path d="M578.2144 270.976c-18.8288 41.6384-83.7888 72.6016-162.4576 72.6016h-47.7824c1.0496 47.1424 5.44 83.7248 23.7312 120.3072 24.064 48.128 73.1648 96.2432 120.3072 96.2432S608.256 512 632.32 463.8848c21.4272-42.8416 23.7696-85.696 24.0256-145.5232-1.4208-27.0464-52.48-47.3856-78.1312-47.3856z" fill="#F6CBAD" p-id="4279"></path><path d="M723.6864 283.8912c-21.0176-75.904-107.8016-132.8-211.6864-132.8s-190.6688 56.896-211.6864 132.8c-16.0512 8.8064-28.928 21.504-28.928 35.6224v48.128c0 26.5728 45.6064 48.128 72.1792 48.128v-96.2432c0-66.4448 75.4048-120.3072 168.4352-120.3072s168.4352 53.8624 168.4352 120.3072v48.128c0 51.5712-32.448 95.552-78.0288 112.6656-6.4384-9.8816-17.5872-16.4224-30.2464-16.4224h-48.128c-19.9296 0-36.096 16.1664-36.096 36.096 0 19.9296 16.1664 36.096 36.096 36.096H572.16c3.52 0 6.912-0.512 10.1248-1.4464 70.9888-9.3312 128.0512-62.8608 142.6432-132.0576 15.4752-8.768 27.6992-21.1712 27.6992-34.9184v-48.128c-0.0128-14.144-12.8896-26.8416-28.9408-35.648z" fill="#4D4D4D" p-id="4280"></path></svg>'

prompt_default= __('''
你是一个AI助手，以下```中的内容是你已知的知识。
```
{{knowledge}}
```

你的任务是根据上面给出的知识，回答用户的问题。当你回答时，你的回复必须遵循以下约束：

1. 只回复以上知识中包含的信息。
2. 当你回答问题需要一些额外知识的时候，只能使用非常确定的知识和信息，以确保不会误导用户。
3. 如果你无法确切回答用户问题的答案，请直接回复"不知道"，并给出原因。
4. 使用中文回答。

你需要回答：

{{query}}
'''.strip())

class Chat_View_Base():
    datamodel = SQLAInterface(Chat)
    route_base = '/chat_modelview/api'
    label_title = _('聊天窗配置')
    base_order = ("id", "desc")
    order_columns = ['id']
    base_filters = [["id", Chat_Filter, lambda: []]]  # 设置权限过滤器

    spec_label_columns = {
        "chat_type": _("聊天窗类型"),
        "hello": _("欢迎语"),
        "tips": _("输入示例"),
        "knowledge": _("知识库"),
        "service_type": _("接口类型"),
        "service_config": _("接口配置"),
        "session_num": _("上下文条数"),
        "prompt": _("提示词模板")
    }

    list_columns = ['name', 'icon', 'label', 'chat_type', 'service_type', 'owner', 'session_num', 'hello']
    cols_width = {
        "name": {"type": "ellip1", "width": 100},
        "label": {"type": "ellip2", "width": 150},
        "chat_type": {"type": "ellip1", "width": 100},
        "hello": {"type": "ellip1", "width": 200},
        "tips": {"type": "ellip1", "width": 200},
        "service_type": {"type": "ellip1", "width": 100},
        "owner": {"type": "ellip1", "width": 200},
        "session_num":{"type": "ellip1", "width": 100},
        "knowledge": {"type": "ellip1", "width": 200},
        "prompt": {"type": "ellip1", "width": 200},
        "service_config": {"type": "ellip1", "width": 200},
    }
    default_icon = '<svg t="1683877543698" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4469" width="50" height="50"><path d="M894.1 355.6h-1.7C853 177.6 687.6 51.4 498.1 54.9S148.2 190.5 115.9 369.7c-35.2 5.6-61.1 36-61.1 71.7v143.4c0.9 40.4 34.3 72.5 74.7 71.7 21.7-0.3 42.2-10 56-26.7 33.6 84.5 99.9 152 183.8 187 1.1-2 2.3-3.9 3.7-5.7 0.9-1.5 2.4-2.6 4.1-3 1.3 0 2.5 0.5 3.6 1.2a318.46 318.46 0 0 1-105.3-187.1c-5.1-44.4 24.1-85.4 67.6-95.2 64.3-11.7 128.1-24.7 192.4-35.9 37.9-5.3 70.4-29.8 85.7-64.9 6.8-15.9 11-32.8 12.5-50 0.5-3.1 2.9-5.6 5.9-6.2 3.1-0.7 6.4 0.5 8.2 3l1.7-1.1c25.4 35.9 74.7 114.4 82.7 197.2 8.2 94.8 3.7 160-71.4 226.5-1.1 1.1-1.7 2.6-1.7 4.1 0.1 2 1.1 3.8 2.8 4.8h4.8l3.2-1.8c75.6-40.4 132.8-108.2 159.9-189.5 11.4 16.1 28.5 27.1 47.8 30.8C846 783.9 716.9 871.6 557.2 884.9c-12-28.6-42.5-44.8-72.9-38.6-33.6 5.4-56.6 37-51.2 70.6 4.4 27.6 26.8 48.8 54.5 51.6 30.6 4.6 60.3-13 70.8-42.2 184.9-14.5 333.2-120.8 364.2-286.9 27.8-10.8 46.3-37.4 46.6-67.2V428.7c-0.1-19.5-8.1-38.2-22.3-51.6-14.5-13.8-33.8-21.4-53.8-21.3l1-0.2zM825.9 397c-71.1-176.9-272.1-262.7-449-191.7-86.8 34.9-155.7 103.4-191 190-2.5-2.8-5.2-5.4-8-7.9 25.3-154.6 163.8-268.6 326.8-269.2s302.3 112.6 328.7 267c-2.9 3.8-5.4 7.7-7.5 11.8z" fill="#2c2c2c" p-id="4470"></path></svg>'

    knowledge_config = '''
{
    "type": "api|file",
    "url":"",   # api请求地址
    "headers": {},  # api请求的附加header
    "data": {},   # api请求的附加data
    "file":"/mnt/$username/",   # 文件地址，或者目录地址，可以多个文件
    "upload_url": "", # 知识库的上传地址
    "recall_url": "", # 召回地址
}
    '''
    service_config = '''
{
    "llm_url": "",  # 请求的url
    "llm_headers": {
        "xxxxx": "xxxxxx"   # 额外添加的header
    },
    "llm_tokens": [],    # chatgpt的token池
    "llm_data": {
        "xxxxx": "xxxxxx"   # 额外添加的json参数
    },
    "model_name": "" # 模型名称
    "temperature": 1,  # 多样性，0~2，越大越有创造性
    "top_p":0.5,     # 采样范围，0~1，越低越有变化
    "presence_penalty": 1，   # 词汇控制，0~2，越低越谈论新话题
    "max_tokens": 1500,    # token数目

    "stream": "false" # 是否流式响应
}
        '''

    options_demo={"xAxis":{"type":"category","data":["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]},"yAxis":{"type":"value"},"series":[{"data":[150,230,224,218,135,147,260],"type":"line"}]}
    test_api_resonse = {
        # "text":"这里是文本响应体",
        "echart": json.dumps(options_demo)
    }

    add_fieldsets = [
        (
            _('基础配置'),
            {"fields": ['name','icon','label','doc','owner'], "expanded": True},
        ),
        (
            _('提示词配置'),
            {"fields": ['chat_type','hello','tips','knowledge','prompt','session_num'], "expanded": True},
        ),
        (
            _('模型服务'),
            {"fields": ['service_type','service_config','expand'], "expanded": True},
        )
    ]

    edit_fieldsets=add_fieldsets

    add_form_extra_fields = {
        "name": StringField(
            label= _('名称'),
            description= _('英文名(小写字母、数字、- 组成)，最长50个字符'),
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$")]
        ),
        "label": StringField(
            label= _('标签'),
            default='',
            description = _('中文名'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "icon": StringField(
            label= _('图标'),
            default=default_icon,
            description= _('svg格式图标，图标宽高设置为50*50，<a target="_blank" href="https://www.iconfont.cn/">iconfont</a>'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "owner": StringField(
            label= _('责任人'),
            default='*',
            description= _('可见用户，*表示所有用户可见，将责任人列为第一管理员，逗号分割多个责任人'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

        "chat_type": SelectField(
            label= _('对话类型'),
            description='',
            default='text',
            widget=MySelect2Widget(),
            choices=[['text', _('文本对话')], ['digital_human', _("数字人")]],
            validators=[]
        ),
        "service_type": SelectField(
            label= _('服务类型'),
            description= _('接口类型，并不一定是openai，只需要符合http请求响应格式即可'),
            widget=Select2Widget(),
            default='openai',
            choices=[[x, x] for x in ["chatgpt3.5",'chatgpt4','chatglm','belle', 'llama','AIGC','autogpt',_('召回列表')]],
            validators=[]
        ),
        "service_config": StringField(
            label= _('接口配置'),
            default=json.dumps({
                "llm_url": "",
                "llm_tokens": [],
                "stream": "true"
            },indent=4,ensure_ascii=False),
            description= _('接口配置，每种接口类型配置参数不同'),
            widget=MyBS3TextAreaFieldWidget(rows=5, tips=Markup('<pre><code>' + service_config + "</code></pre>")),
            validators=[DataRequired()]
        ),
        "knowledge": StringField(
            label= _('知识库'),
            default=json.dumps({
                "type": "file",
                "file": [__("文件地址")]
            },indent=4,ensure_ascii=False),
            description= _('先验知识配置。如果先验字符串少于1800个字符，可以直接填写字符串，否则需要使用json配置'),
            widget=MyBS3TextAreaFieldWidget(rows=5, tips=Markup('<pre><code>' + knowledge_config + "</code></pre>")),
            validators=[]
        ),
        "prompt":StringField(
            label= _('提示词'),
            default=prompt_default,
            description= _('提示词模板，包含{{knowledge}}知识库召回内容，{{history}}为多轮对话，{{query}}为用户的问题'),
            widget=MyBS3TextAreaFieldWidget(rows=5),
            validators=[]
        ),
        "tips": StringField(
            label= _('输入示例'),
            default='',
            description= _('提示输入，多个提示输入，多行配置'),
            widget=MyBS3TextAreaFieldWidget(rows=3),
            validators=[]
        ),
        "expand": StringField(
            label= _('扩展'),
            default=json.dumps({
                "index":int(time.time())
            },indent=4,ensure_ascii=False),
            description= _('配置扩展参数，"index":控制显示顺序,"isPublic":控制是否为公共应用'),
            widget=MyBS3TextAreaFieldWidget(),
            validators=[]
        ),
    }
    from copy import deepcopy
    edit_form_extra_fields = add_form_extra_fields

    # @pysnooper.snoop()
    def pre_update_web(self, chat=None):
        pass
        self.edit_form_extra_fields['name'] = StringField(
            _('名称'),
            description=_('英文名(小写字母、数字、- 组成)，最长50个字符'),
            default='',
            widget=MyBS3TextFieldWidget(readonly=True if chat else False),
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$")]
        )

    def pre_add_web(self):
        self.default_filter = {
            "expand": '"isPublic": true'
        }
        self.pre_update_web()

    # 如果传上来的有文件
    # @pysnooper.snoop()
    def pre_add_req(self, req_json=None):

        expand = json.loads(req_json.get('expand','{}'))

        # 私有项目做特殊处理
        if not expand.get('isPublic',True):
            name = f'{g.user.username}-faq-{uuid.uuid4().hex[:4]}'
            req_json['name'] = name
            req_json['hello']= __('自动为您创建的私人对话，不使用上下文，左下角可以清理会话和修改知识库配置')
            req_json['session_num']='0'
            req_json['icon'] = default_icon

            if req_json and 'files' in req_json:
                files_path = []
                files = req_json['files']
                if type(files) != list:
                    files = [files]
                id = req_json.get('id', '')
                exist_knowledge = {}
                if id:
                    chat = db.session.query(Chat).filter_by(id=id).first()
                    if chat:
                        try:
                            exist_knowledge = json.loads(chat.knowledge)
                        except:
                            exist_knowledge = {}

                file_arr = []
                for file in files:
                    file_name = file.get('name', '')
                    file_type = file.get("type", '')
                    file_content = file.get("content", '')   # 最优最新一次上传的才有这个。
                    file_arr.append({
                        "name": file_name,
                        "type": file_type
                    })
                    # 拼接文件保存路径
                    file_path = f'/data/k8s/kubeflow/global/knowledge/{name}/{file_name}'
                    files_path.append(file_path)
                    # 如果有新创建的文件内容
                    if file_content:
                        content = base64.b64decode(file_content)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        file = open(file_path, mode='wb')
                        file.write(content)
                        file.close()


                knowledge = {
                    "type": "file",
                    "file": files_path,
                    # "file_arr": file_arr,
                    "upload_url": "http://chat-embedding.aihub:80/aihub/chat-embedding/api/upload_files",
                    "recall_url": "http://chat-embedding.aihub:80/aihub/chat-embedding/api/recall"
                }
                expand['fileMetaList']=file_arr

                if exist_knowledge.get('status',''):
                    knowledge['status']=exist_knowledge.get('status','')
                    knowledge['update_time'] = exist_knowledge.get('update_time','')

                req_json['knowledge'] = json.dumps(knowledge, indent=4, ensure_ascii=False)
                req_json['expand']=json.dumps(expand, indent=4, ensure_ascii=False)
                del req_json['files']

        return req_json

    pre_update_req = pre_add_req

    # @pysnooper.snoop(watch_explode=('req_json',))
    # def pre_update_req(self, req_json=None):
    #     print(g.user.username)
    #     owner = req_json.get('owner','')
    #     if g.user.username in owner:
    #         self.pre_add_req(req_json)
    #     else:
    #         flash('只有创建者或管理员可以配置', 'warning')
    #         raise MyappException('just creator can add/edit')

    # @pysnooper.snoop()
    def pre_add(self, item):
        if not item.knowledge or not item.knowledge.strip():
            item.knowledge = '{}'
        if not item.owner:
            item.owner = g.user.username
        if not item.icon:
            item.icon = default_icon
        if not item.chat_type:
            item.chat_type = 'text'
        if not item.service_type:
            item.service_type = 'openai'
        if not item.service_config or not item.service_config.strip():
            service_config = {
                "llm_url": "",
                "llm_tokens": [],
                "stream": "true"
            }
            item.service_config = json.dumps(service_config)

        service_config = json.loads(item.service_config) if item.service_config.strip()else {}
        expand = json.loads(item.expand) if item.expand.strip() else {}
        knowledge = json.loads(item.knowledge) if item.knowledge.strip() else {}

        # 配置扩展字段
        if item.expand.strip():
            item.expand=json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        try:
            expand = json.loads(item.expand)
            expand['isPublic'] = expand.get('isPublic',True)
            # 把之前的属性更新上，避免更新的时候少填了什么属性
            src_expand = self.src_item_json.get("expand",'{}')
            if src_expand:
                src_expand = json.loads(src_expand)
                src_expand.update(expand)
                expand = src_expand
            item.expand = json.dumps(expand, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)

        # 如果是私有应用，添加一些file_arr
        if not expand.get('isPublic', True):
            fileMetaList = expand.get('fileMetaList',[])
            files = knowledge.get('file',[])
            # 如果有文件，但是没有文件属性信息，则更新
            if not fileMetaList and files:
                expand['fileMetaList'] = []
                if type(files)!=list:
                    files = [files]
                for file in files:
                    name = os.path.basename(file)

                    if '.' in name:
                        ext = name[name.rindex('.')+1:]
                        file_map={
                            "map":"application/octet-stream",
                            "csv":"text/csv",
                            "pdf":"application/pdf",
                            "txt":"text/plain"
                        }
                        file_attr = {
                            "name": name,
                            "type": file_map[ext]
                        }
                        expand['fileMetaList'].append(file_attr)



            # 如果有知识库
            if knowledge.get('file',[]) or knowledge.get('url',''):
                item.prompt = prompt_default
            # 如果没有，就自动多轮对话
            else:
                knowledge['status']='在线'
                knowledge['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                item.session_num=10
                item.prompt = '''
{{history}}
Human:{{query}}
AI:
'''.strip()


        item.icon = item.icon.replace('width="200"','width="50"').replace('height="200"','height="50"')
        if '{{query}}' not in item.prompt:
            item.prompt = item.prompt+"\n{{query}}\n"

        item.service_config = json.dumps(service_config, indent=4, ensure_ascii=False)
        item.expand = json.dumps(expand, indent=4, ensure_ascii=False)
        try:
            item.knowledge = json.dumps(knowledge, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)

    # @pysnooper.snoop()
    def pre_update(self, item):
        if g.user.username in self.src_item_json.get('owner','') or g.user.is_admin():
            self.pre_add(item)
        else:
            flash(__('只有创建者或管理员可以配置'), 'warning')
            raise MyappException('just creator can add/edit')

    def post_add(self, item):
        try:
            if not self.src_item_json:
                self.src_item_json = {}

            src_file = json.loads(self.src_item_json.get('knowledge', '{}')).get("file", '')
            last_time = json.loads(self.src_item_json.get('knowledge', '{}')).get("update_time",'')
            if last_time:
                last_time = datetime.datetime.strptime(last_time,'%Y-%m-%d %H:%M:%S')

            knowledge_config = json.loads(item.knowledge) if item.knowledge else {}
            exist_file = knowledge_config.get("file", '')
            # 文件变了，或者更新时间过期了，都要重新更新
            if exist_file and src_file != exist_file or not last_time or (datetime.datetime.now()-last_time).total_seconds()>3600:
                self.upload_knowledge(chat=item, knowledge_config=knowledge_config)

        except Exception as e:
            print(e)

    def post_update(self, item):
        self.post_add(item)

    # 按配置的索引进行排序
    def post_list(self, items):
        from myapp.utils import core
        return core.sort_expand_index(items)
        # print(_response['data'])
        # _response['data'] = sorted(_response['data'],key=lambda chat:float(json.loads(chat.get('expand','{}').get("index",1))))

    @action("copy", "复制", confirmation= '复制所选记录?', icon="fa-copy", multiple=True, single=True)
    def copy(self, chats):
        if not isinstance(chats, list):
            chats = [chats]
        try:
            for chat in chats:
                new_chat = chat.clone()
                new_chat.name = new_chat.name+"-copy"
                new_chat.created_on = datetime.datetime.now()
                new_chat.changed_on = datetime.datetime.now()
                db.session.add(new_chat)
                db.session.commit()
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            print(e)
            raise e

        return redirect(request.referrer)

    @expose('/chat/<chat_name>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def chat(self, chat_name, args=None):
        if chat_name == 'chatbi':
            files = os.listdir('myapp/utils/echart/')
            files = ['area-stack.json', 'rose.json', 'mix-line-bar.json', 'pie-nest.json', 'bar-stack.json',
                   'candlestick-simple.json', 'graph-simple.json', 'tree-polyline.json', 'sankey-simple.json',
                   'radar.json', 'sunburst-visualMap.json', 'parallel-aqi.json', 'funnel.json',
                   'sunburst-visualMap.json', 'scatter-effect.json']
            files = [os.path.join('myapp/utils/echart/',file) for file in files if '.json' in file]

            return {
                "status": 0,
                "finish": False,
                "message": 'success',
                "result": [
                    {
                        "text":"未配置后端模型，这里生成示例看板\n\n",
                        # "echart": json.dumps(options_demo)
                        "echart":open(random.choice(files)).read()
                    }
                ]
            }

        if not args:
            args = request.get_json(silent=True)
        if not args:
            args = {}
        session_id = args.get('session_id', 'xxxxxx')
        request_id = args.get('request_id', str(datetime.datetime.now().timestamp()))
        search_text = args.get('search_text', '')
        search_audio = args.get('search_audio', None)
        search_image = args.get('search_image', None)
        search_video = args.get('search_video', None)
        username = args.get('username', '')
        enable_tts = args.get('enable_tts', False)
        if not username:
            username = g.user.username
        if g:
            g.after_message=''
        stream = args.get('stream', False)
        if str(stream).lower()=='false':
            stream = False

        begin_time = datetime.datetime.now()

        chat = db.session.query(Chat).filter_by(name=chat_name).first()
        if not chat:
            return jsonify({
                "status": 1,
                "message": __('聊天不存在'),
                "result": []
            })

        # 如果超过一定聊天数目，则禁止
        # if username not in conf.get('ADMIN_USER').split(','):
        #     log_num = db.session.query(ChatLog).filter(ChatLog.username==username).filter(ChatLog.answer_status=='成功').filter(ChatLog.created_on>datetime.datetime.now().strftime('%Y-%m-%d')).all()
        #     if len(log_num)>10:
        #         return jsonify({
        #             "status": 1,
        #             "finish": 0,
        #             "message": '聊天次数达到上限，每人，每天仅限10次',
        #             "result": [{"text":"聊天次数达到上限，每人，每天仅限10次"}]
        #         })

        stream_config = json.loads(chat.service_config).get('stream', True)
        if stream_config==False or str(stream_config).lower() == 'false':
            stream = False

        enable_history = args.get('history', True)
        chatlog=None
        # 添加数据库记录
        try:
            text = emoji.demojize(search_text)
            search_text = re.sub(':\S+?:', ' ', text)  # 去除表情
            chatlog = ChatLog(
                username=str(username),
                chat_id=chat.id,
                query=search_text,
                answer="",
                manual_feedback="",
                answer_status="created",
                answer_cost='0',
                err_msg="",
                created_on=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                changed_on=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            )
            db.session.add(chatlog)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(e)

        return_result = []
        return_message = ''
        return_status = 1
        finish = ''
        answer_status = 'making'
        err_msg = ''
        history = []
        try:
            if enable_history and int(chat.session_num):
                history = cache.get('chat_' + session_id)  # 所有需要的上下文
                if not history:
                    history = []
        except Exception as e:
            print(e)

        if chat.service_type.lower() == 'openai' or chat.service_type.lower() == 'chatgpt4' or chat.service_type.lower() == 'chatgpt3.5':
            if stream:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()
                res = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=True
                )
                if chatlog:
                    chatlog.answer_status = '成功'
                    db.session.commit()
                return res

            else:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()

                return_status, text = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=False
                )
                return_message = __('失败') if return_status else __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    }
                ]


        if chat.service_type.lower() == 'openai' or chat.service_type.lower() == 'chatgpt4' or chat.service_type.lower() == 'chatgpt3.5':
            if stream:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()
                res = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=True
                )
                if chatlog:
                    chatlog.answer_status = __('成功')
                    db.session.commit()
                return res

            else:
                if chatlog:
                    chatlog.answer_status = 'push chatgpt'
                    db.session.commit()

                return_status, text = self.chatgpt(
                    chat=chat,
                    session_id=session_id,
                    search_text=search_text,
                    enable_history=enable_history,
                    history=history,
                    chatlog_id=chatlog.id,
                    stream=False
                )
                return_message = __('失败') if return_status else __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    }
                ]

        # 仅返回召回列表
        if __('召回列表') in chat.service_type.lower():
            knowledge = self.get_remote_knowledge(chat, search_text,score=True)
            knowledge = [__("内容：\n\n    ") + x['context'].replace('\n','\n    ') + "\n\n" + __("得分：\n\n    ") + str(x.get('score', '')) + "\n\n" + __("文件：\n\n    ") + str(x.get('file', '')) for x in knowledge]
            if knowledge:
                text = '\n\n-------\n'.join(knowledge)
                return_message = __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    }
                ]
            else:
                return_result = [
                    {
                        "text": __('召回内容为空')
                    }
                ]

        # 多轮召回方式
        if __('多轮') in chat.service_type.lower():
            knowledge = self.get_remote_knowledge(chat, search_text)
            if knowledge:
                return_message = __("成功")
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    } for text in knowledge
                ]
            else:
                return_result = [
                    {
                        "text": __('未找到相关内容')
                    }
                ]

        if chat.service_type.lower() == 'aigc':
            return_status, return_res = self.aigc4(chat=chat, search_text=search_text)
            if not return_status:
                return return_res

        # 添加数据库记录
        if chatlog:
            try:
                canswar = "\n".join(item.get('text','') for item in return_result)
                chatlog.query = search_text
                chatlog.answer = canswar
                chatlog.answer_cost = str((datetime.datetime.now()-begin_time).total_seconds())
                chatlog.answer_status=answer_status,
                chatlog.err_msg = return_message
                db.session.commit()

                # 正确响应的话，才设置为历史状态
                if history != None and not return_status:
                    history.append((search_text, canswar))
                    history = history[0 - int(chat.session_num):]
                    try:
                        cache.set('chat_' + session_id, history, timeout=300)   # 人连续对话的时间跨度
                    except Exception as e:
                        print(e)

            except Exception as e:
                db.session.rollback()
                print(e)

        return {
            "status": return_status,
            "finish": finish,
            "message": return_message,
            "result": [x for x in return_result if x]
        }

    # @pysnooper.snoop()
    def upload_knowledge(self,chat,knowledge_config):
        """
        上传文件到远程服务
        @param chat: 场景对象
        @param knowledge_config: 知识库配置
        @return:
        """
        # 没有任何值就是空的
        files=[]
        if not knowledge_config:
            return ''

        knowledge_type = knowledge_config.get("type", 'str')
        if knowledge_type == 'str':
            knowledge = knowledge_config.get("content", '')
            if knowledge:
                file_path = f'knowledge/{chat.name}/{str(time.time()*1000)}'
                os.makedirs(os.path.dirname(file_path),exist_ok=True)
                file = open(file_path,mode='w')
                file.write(knowledge)
                file.close()
                files.append(file_path)

        if knowledge_type == 'api':
            url = knowledge_config.get("url", '')
            if not url:
                return ''
            headers = knowledge_config.get("headers", {})
            data = knowledge_config.get("data", {})
            if data:
                res = requests.post(url, headers=headers, json=data,verify=False)
            else:
                res = requests.get(url, headers=headers,verify=False)

            if res.status_code == 200:
                # 获取文件名和文件格式
                filename = os.path.basename(url)
                file_format = os.path.splitext(filename)[1]

                # 拼接文件保存路径
                file_path = f'knowledge/{chat.name}/{str(time.time() * 1000)}'
                if file_format:
                    file_path = file_path+"."+file_format
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file = open(file_path, mode='wb')
                file.write(res.content)
                file.close()
                files.append(file_path)

        if knowledge_type == 'file':
            file_paths = knowledge_config.get("file", '')
            if type(file_paths)!=list:
                file_paths = [file_paths]
            for file_path in file_paths:
                if re.match('^/mnt', file_path):
                    file_path = "/data/k8s/kubeflow/pipeline/workspace" + file_path.replace("/mnt", '')
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        files.append(file_path)
                    if os.path.isdir(file_path):
                        for root, dirs_temp, files_temp in os.walk(file_path):
                            for name in files_temp:
                                one_file_path = os.path.join(root, name)
                                # print(one_file_path)
                                if os.path.isfile(one_file_path):
                                    files.append(one_file_path)

        if knowledge_type == 'sql':
            return ''


        service_config = json.loads(chat.service_config)
        upload_url = knowledge_config.get("upload_url", '')
        if files:
            if '127.0.0.1' in request.host_url:
                print('发现的知识库文件：',files)
                knowledge = json.loads(chat.knowledge) if chat.knowledge else {}
                knowledge['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                knowledge['status'] = "uploading"
                chat.knowledge = json.dumps(knowledge,indent=4,ensure_ascii=False)
                db.session.commit()
                files_content = [('files', (os.path.basename(file), open(file, 'rb'))) for file in files]
                data = {"chat_id": chat.name}
                response = requests.post(upload_url, files=files_content, data=data,verify=False)
                print('上传私有知识响应：',json.dumps(json.loads(response.text), ensure_ascii=False, indent=4))
                knowledge['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                knowledge['status'] = "online"
                chat.knowledge = json.dumps(knowledge, indent=4, ensure_ascii=False)
                db.session.commit()

            else:
                from myapp.tasks.async_task import upload_knowledge
                kwargs = {
                    "files": files,
                    "chat_id":chat.name,
                    "upload_url":upload_url,
                    # 可以根据不同的配置来决定对数据做什么处理，比如
                    # "config":{
                    #     "file":{
                    #         "cube-studio.csv":{
                    #             "embedding_columns": ["问题"],
                    #             "llm_columns": ['问题', '答案'],
                    #             "keywork_columns": [],
                    #             "summary_columns": []
                    #         }
                    #     }
                    # }
                }
                upload_knowledge.apply_async(kwargs=kwargs)

    all_chat_knowledge = {}
    # 根据配置获取远程的先验知识
    # @pysnooper.snoop()
    def get_remote_knowledge(self,chat,search_text,score=False):
        """
        召回服务
        @param chat: 场景对象
        @param search_text: 搜索文本
        @return: 获取召回的前3个文本
        """
        knowledge=[]
        try:
            service_config = json.loads(chat.service_config)
            knowledge_config = json.loads(chat.knowledge)

            # 时间过时就发过去重新更新知识库
            update_time = knowledge_config.get("update_time",'')
            if update_time:
                update_time = datetime.datetime.strptime(update_time,'%Y-%m-%d %H:%M:%S')
            if not update_time or (datetime.datetime.now()-update_time).total_seconds()>3600 or knowledge_config.get("status","")!='在线':
                self.upload_knowledge(chat=chat,knowledge_config=knowledge_config)


            # 进行召回
            recall_url = knowledge_config.get("recall_url", '')
            if recall_url:
                data={
                    "knowledge_base_id":chat.name,
                    "question":search_text,
                    "history":[]
                }
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                res = requests.post(recall_url,json=data,headers=headers,timeout=5,verify=False)
                if res.status_code==200:
                    recall_result = res.json()
                    print('召回响应',json.dumps(recall_result,indent=4,ensure_ascii=False))
                    if 'result' in recall_result:
                        knowledge= recall_result['result']
                        if not score:
                            knowledge = [x['context'] for x in knowledge]
                        knowledge = knowledge[0:3]
                    else:
                        source_documents = recall_result.get('source_documents',[])
                        # source_documents = sorted(source_documents,key=lambda item:float(item.get('score',1)))  # 按分数值排序，应该走排序算法
                        source_documents = source_documents[:3]  # 只去前面3个
                        all_sources = []
                        for index,item in enumerate(source_documents):
                            if int(item.get('score', 1)) > float(knowledge_config.get("min_score",0)):   # 根据最小分数值确定
                                source = item.get('source', '')
                                if source:
                                    all_sources.append(source)

                                knowledge.append(item['context'])
                        all_sources = [x.strip() for x in list(set(all_sources)) if x.strip()]
                        after_message = ''
                        if all_sources:
                            for index,source in enumerate(all_sources):
                                source_url = request.host_url.rstrip('/') + f"/aitalk_modelview/api/file/{chat.name}/" + source.lstrip('/')
                                after_message += f'[文档{index}]({source_url}) '
                            # g.after_message = g.after_message + f"\n\n{after_message}"

        except Exception as e:
            print(e)

        return knowledge

    # @pysnooper.snoop()
    # 获取header和url
    def get_llm_url_header(self,chat,stream=False):
        """
        获取访问地址和有效token
        @param chat:
        @param stream:
        @return:
        """
        url = json.loads(chat.service_config).get("llm_url", '')
        headers = json.loads(chat.service_config).get("llm_headers", {})
        if stream:
            headers['Accept'] = 'text/event-stream'
        else:
            headers['Accept'] = 'application/json'

        if not url:
            llm_url = conf.get('CHATGPT_CHAT_URL', 'https://api.openai.com/v1/chat/completions')
            if llm_url:
                if type(llm_url) == list:
                    llm_url = random.choice(llm_url)
                else:
                    llm_url = llm_url
                url=llm_url

        llm_tokens = json.loads(chat.service_config).get("llm_tokens", [])
        llm_token = ''
        if llm_tokens:
            if type(llm_tokens) != list:
                llm_tokens = [llm_tokens]
            # 如果有过多错误的token，则直接废弃
            error_token = json.loads(chat.service_config).get("miss_tokens",{})
            if error_token:
                right_llm_tokens= [token for token in llm_token if int(error_token.get(token,0))<100]
                if right_llm_tokens:
                    llm_tokens=right_llm_tokens

            llm_token = random.choice(llm_tokens)
            headers['Authorization'] = 'Bearer ' + llm_token  # openai的接口
            headers['api-key'] = llm_token   # 微软的接口
        else:
            llm_tokens = conf.get('CHATGPT_TOKEN','')
            if llm_tokens:
                if type(llm_tokens)==list:
                    llm_token = random.choice(llm_tokens)
                else:
                    llm_token = llm_tokens
                headers['Authorization'] = 'Bearer ' + llm_token    # openai的接口
                headers['api-key']=llm_token    # 微软的接口

        return url,headers,llm_token

    # 组织提问词
    # @pysnooper.snoop(watch_explode=('system_content'))
    def generate_prompt(self,chat, search_text, enable_history, history=[]):
        messages = chat.prompt

        messages = messages.replace('{{query}}', search_text)

        # 获取知识库
        if '{{knowledge}}' in chat.prompt:
            knowledge = chat.knowledge  # 直接使用原文作为知识库
            try:
                knowledge_config = json.loads(chat.knowledge)
                try:
                    knowledge = self.get_remote_knowledge(chat, search_text)
                except Exception as e1:
                    print(e1)
            except Exception as e:
                print(e)

            if type(knowledge) != list:
                knowledge = [str(knowledge)]
            knowledge = [x for x in knowledge if x.strip()]
            # 拼接请求体
            print('召回知识库', json.dumps(knowledge, indent=4, ensure_ascii=False))
            added_knowledge = []
            # 添加私有知识库，要满足token限制
            for item in knowledge:
                # 至少要保留前置语句，后置语句，搜索语句。
                if sum([len(x) for x in added_knowledge]) < (max_len - len(messages) - len(item)):
                    added_knowledge.append(item)
            added_knowledge = '\n\n'.join(added_knowledge)
            messages = messages.replace('{{knowledge}}', added_knowledge)

        if '{{history}}' in chat.prompt:

            # 拼接上下文
            # 先从后往前加看看个数是不是超过了门槛
            added_history=[]
            if enable_history and history:
                for index in range(len(history) - 1, -1, -1):
                    faq = history[index]
                    added_faq="Human: %s\nAI: %s"%(faq[0],faq[1])
                    added_history_len = sum([len(x) for x in added_faq])
                    if len(added_faq) < (max_len-len(messages)-added_history_len):
                        added_history.insert(0,added_faq)
                    else:
                        break
            added_history = '\n'.join(added_history)
            messages = messages.replace('{{history}}', added_history)
        print(messages)
        return [{'role': 'user', 'content': messages}]

    # 生成openai相应格式
    def make_openai_res(self,message,stream=True):
        back = {
            "id": "chatcmpl-7OPUNz80uRGVKLcBMW8aKZT9dg938",
            "object": "chat.completion.chunk" if stream else 'chat.completion',
            "created": int(time.time()),
            "model": "gpt-3.5-turbo-16k-0613",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "role": "assistant",
                        "content":message
                    },
                    "message":{
                        "role": "assistant",
                        "content": message
                    }
                }
            ],
            "usage": None
        }
        return json.dumps(back)

    @expose('/chat/chatgpt/<chat_name>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def chatgpt_api(self, chat_name):
        """
        为调用chatgpt单独提供的接口
        @param chat_name:
        @return:
        """
        args = request.get_json(silent=True)
        chat = db.session.query(Chat).filter_by(name=chat_name).first()

        session_id = args.get('session_id', 'xxxxxx')
        request_id = args.get('request_id', str(datetime.datetime.now().timestamp()))
        search_text = args.get('search_text', '')

        return_status, text = self.chatgpt(
            chat=chat,
            session_id=session_id,
            search_text=search_text,
            enable_history=False,
            history=[],
            chatlog_id=None,
            stream=False
        )
        return jsonify({
            "status": return_status,
            "message": __('失败') if return_status else __("成功"),
            "result": [
                {
                    "text": text
                }
            ]
        })

    # 调用chatgpt接口
    # @pysnooper.snoop()
    def chatgpt(self, chat, session_id, search_text, enable_history,history=[], chatlog_id=None, stream=True):
        max_retry=3
        for i in range(0,max_retry):

            url, headers, llm_token = self.get_llm_url_header(chat, stream)
            message = self.generate_prompt(chat=chat, search_text=search_text, enable_history=enable_history, history=history)
            service_config = json.loads(chat.service_config)
            data = {
                'model': 'gpt-3.5-turbo-16k-0613',
                'messages': message,
                'temperature': service_config.get("temperature",1),  # 问答发散度 0-2 越高越发散 较高的值（如0.8）将使输出更随机，较低的值（如0.2）将使其更集中和确定性
                'top_p': service_config.get("top_p",0.5),  # 同temperature，如果设置 0.1 意味着只考虑构成前 10% 概率质量的 tokens
                'n': 1,  # top n可选值
                'stream': stream,
                'stop': 'elit proident sint',  #
                'max_tokens': service_config.get("max_tokens",1500),  # 最大返回数
                'presence_penalty': service_config.get("presence_penalty",1),  # [控制主题的重复度]，-2.0（抓住一个主题使劲谈论） ~ 2.0（最大程度避免谈论重复的主题） 之间的数字，正值会根据到目前为止是否出现在文本中来惩罚新 tokens，从而增加模型谈论新主题的可能性
                'frequency_penalty': 0, # [重复度惩罚因子], -2.0(可以尽情出现相同的词汇) ~ 2.0 (尽量不要出现相同的词汇)
                'user': 'user',
            }
            data.update(json.loads(chat.service_config).get("llm_data", {}))
            if stream:
                # 返回流响应
                import sseclient

                res = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    stream=stream,
                    verify=False
                )
                if res.status_code != 200 and i<(max_retry-1):
                    continue
                client = sseclient.SSEClient(res)

                # @pysnooper.snoop()
                def generate(history):

                    back_message = ''
                    for event in client.events():
                        message = event.data
                        finish = False
                        if message != '[DONE]':
                            message = json.loads(event.data)['choices'][0].get('delta', {}).get('content', '')
                            print(message, flush=True, end='')
                        # print(message)
                        if message == '[DONE]':
                            finish = True
                            back_message = back_message+g.after_message
                            if chatlog_id:
                                chatlog = db.session.query(ChatLog).filter_by(id=int(chatlog_id)).first()
                                chatlog.answer_status = '成功'
                                chatlog.answer = back_message
                                db.session.commit()
                                if history != None:
                                    history.append((search_text, back_message))
                                    history = history[0 - int(chat.session_num):]
                                    try:
                                        cache.set('chat_' + session_id, history, timeout=300)  # 人连续对话的时间跨度
                                    except Exception as e:
                                        print(e)
                        else:
                            back_message = back_message + message
                        # 随机乱码，用来避免内容中包含此内容，实现每次返回内容的分隔
                        back = "TQJXQKT0POF6P4D:" + json.dumps(
                            {
                                "message": "success",
                                "status": 0,
                                "finish":finish,
                                "result": [
                                    {"text": back_message},
                                ]
                            }, ensure_ascii=False
                        ) + "\n\n"
                        yield back

                response = Response(stream_with_context(generate(history=history if enable_history else None)),mimetype='text/event-stream')
                response.headers["Cache-Control"] = "no-cache"
                response.headers["Connection"] = 'keep-alive'
                response.status_code = res.status_code
                if response.status_code ==401:
                    service_config = json.loads(chat.service_config)
                    # if 'miss_tokens' not in service_config:
                    #     service_config['miss_tokens']={}
                    # service_config['miss_tokens'][llm_token]=service_config['miss_tokens'].get(llm_token,0)+1
                    chat.service_config = json.dumps(service_config,ensure_ascii=False,indent=4)
                    db.session.commit()

                return response

                # 返回普通响应
            else:
                # print(url)
                # print(headers)
                # print(data)
                res = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    verify=False
                )
                if res.status_code != 200 and i < (max_retry - 1):
                    continue
                if res.status_code == 200 or res.status_code == 201:
                    # print(res.text)
                    mes = res.json()['choices'][0]['message']['content']
                    print(mes)
                    return 0, mes
                else:
                    service_config = json.loads(chat.service_config)
                    # if 'miss_tokens' not in service_config:
                    #     service_config['miss_tokens'] = {}
                    # service_config['miss_tokens'][llm_token] = service_config['miss_tokens'].get(llm_token,0) + 1
                    chat.service_config = json.dumps(service_config, ensure_ascii=False, indent=4)
                    db.session.commit()
                    return 1, f'请求{url}失败'

    # @pysnooper.snoop()
    def aigc4(self, chat, search_text):
        """
        aigc 文本转图片
        @param chat:
        @param search_text:
        @return:
        """
        try:
            url = json.loads(chat.service_config).get("aigc_url", '')
            if 'http:' not in url and 'https://' not in url:
                url = urllib.parse.urljoin(request.host_url, url)
            pic_num = json.loads(chat.service_config).get("pic_num", 4)
            headers = json.loads(chat.service_config).get("aigc_headers", {})
            data = {
                "text": search_text,
                "prompt": search_text,
                "steps":50
            }
            data.update(json.loads(chat.service_config).get("aigc_data", {}))
            # @pysnooper.snoop()
            from myapp.utils.core import pic2html
            def generate():
                all_result_image = []
                for i in range(pic_num):
                    # 示例输入
                    time.sleep(1)
                    status, image = 0,f'https://cube-studio.oss-cn-hangzhou.aliyuncs.com/aihub/aigc/aigc{i+1}.jpeg'

                    if not status:
                        all_result_image.append(image)

                    back_message = "未配置后端模型，为您生成4张示例图片：\n"+pic2html(all_result_image,pic_num)
                    # print(back_message)

                    back = "TQJXQKT0POF6P4D:" + json.dumps(
                        {
                            "message": "success",
                            "status": 0,
                            "finish": False,
                            "result": [
                                {"text": back_message},
                            ]
                        }, ensure_ascii=False
                    ) + "\n\n"
                    yield back

            response = Response(stream_with_context(generate()),mimetype='text/event-stream')
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = 'keep-alive'
            return 0,response

        except Exception as e:
            return 1, 'aigc报错：' + str(e)

# 添加api
class Chat_View(Chat_View_Base, MyappModelRestApi):


    datamodel = SQLAInterface(Chat)

# 添加api
class Chat_View_Api(Chat_View_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Chat)
    route_base = '/aitalk_modelview/api'
    list_columns = ['id','name', 'icon', 'label', 'chat_type', 'service_type', 'owner', 'session_num', 'hello', 'tips','knowledge','service_config','expand']

    # info接口响应修正
    # @pysnooper.snoop()
    def pre_list_res(self, _response):

        # 把提示语进行分割
        for chat in _response['data']:
            chat['tips'] = [x for x in chat['tips'].split('\n') if x] if chat['tips'] else []
            try:
                service_config = chat.get('service_config', '{}')
                if service_config:
                    chat['service_config'] = json.loads(service_config)
            except Exception as e:
                print(e)
            try:
                knowledge = chat.get('knowledge', '{}')
                if knowledge:
                    chat['knowledge'] = json.loads(knowledge)
            except Exception as e:
                print(e)
            try:
                expand = chat.get('expand', '{}')
                if expand:
                    chat['expand'] = json.loads(expand)
            except Exception as e:
                print(e)
        return _response

appbuilder.add_api(Chat_View)
appbuilder.add_api(Chat_View_Api)
