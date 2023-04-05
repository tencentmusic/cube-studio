import re

from myapp.models.model_chat import Chat,ChatLog
import requests
import time
from sqlalchemy.exc import InvalidRequestError
import datetime
from flask import Response
from flask_appbuilder import action
from flask_appbuilder.models.sqla.interface import SQLAInterface
from wtforms.validators import DataRequired,Regexp
from myapp import app, appbuilder
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget
from flask import jsonify, Markup, make_response, stream_with_context
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
import threading
import queue
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)

conf = app.config
logging = app.logger
import copy
all_queue={

}
from similarities import Similarity


class ChatSimilarity(Similarity):
    all_vector={}

    def chat_add_corpus(self,chat_id,**kwargs):
        # 清空掉
        self.corpus = {}
        self.corpus_embeddings = []
        self.add_corpus(**kwargs)
        self.all_vector[str(chat_id)]={
            "time":datetime.datetime.now(),   # 更新时间
            "corpus":copy.deepcopy(self.corpus),   # 原始语料
            "corpus_ids_map":copy.deepcopy(self.corpus_ids_map),  # 对应关系
            "corpus_embeddings":copy.deepcopy(self.corpus_embeddings)  # 向量
        }


    def chat_most_similar(self,chat_id,**kwargs):
        self.corpus_embeddings = self.all_vector[str(chat_id)]['corpus_embeddings']
        self.corpus = self.all_vector[str(chat_id)]['corpus']
        self.corpus_ids_map = self.all_vector[str(chat_id)]['corpus_ids_map']
        return self.most_similar(**kwargs)

        pass


chatmodel = None

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
        "tips": "提示语",
        "knowledge": "先验知识",
        "service_type": "推理服务类型",
        "service_config": "推理配置",
        "session_num":"上下文记录个数"
    }

    list_columns = ['name','icon','label', 'chat_type', 'service_type','owner','session_num','hello','tips']
    cols_width = {
        "name": {"type": "ellip1", "width": 100},
        "label": {"type": "ellip2", "width": 200},
        "chat_type": {"type": "ellip1", "width": 100},
        "hello": {"type": "ellip1", "width": 200},
        "tips": {"type": "ellip1", "width": 200},
        "service_type": {"type": "ellip1", "width": 100}
    }
    icon='<svg t="1679629996403" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3590" width="50" height="50"><path d="M291.584 806.314667c-13.909333 0-25.6-11.690667-25.6-25.6v-145.92c0-135.68 110.336-246.016 246.016-246.016s246.016 110.336 246.016 246.016v145.92c0 13.909333-11.690667 25.6-25.6 25.6H291.584z" fill="#64EDAC" p-id="3591"></path><path d="M627.114667 626.517333c-18.773333 0-34.133333-15.36-34.133334-34.133333v-36.096c0-18.773333 15.36-34.133333 34.133334-34.133333s34.133333 15.36 34.133333 34.133333v36.096c0 18.773333-15.36 34.133333-34.133333 34.133333zM396.885333 626.517333c-18.773333 0-34.133333-15.36-34.133333-34.133333v-36.096c0-18.773333 15.36-34.133333 34.133333-34.133333s34.133333 15.36 34.133334 34.133333v36.096c0 18.773333-15.36 34.133333-34.133334 34.133333z" fill="#333C4F" p-id="3592"></path><path d="M735.829333 356.949333l44.288-77.226666c23.552 0.085333 46.506667-12.202667 59.306667-34.389334 19.029333-33.194667 8.106667-75.776-24.490667-95.232-32.512-19.370667-74.325333-8.192-93.354666 24.917334-12.714667 22.186667-11.946667 48.64-0.341334 69.546666l-42.752 74.496a355.1488 355.1488 0 0 0-166.4-41.301333 355.072 355.072 0 0 0-159.744 37.888l-40.704-70.997333a70.8608 70.8608 0 0 0-0.341333-69.546667c-19.029333-33.194667-60.842667-44.373333-93.354667-24.917333-32.512 19.370667-43.52 62.037333-24.490666 95.232a68.027733 68.027733 0 0 0 59.306666 34.389333l41.557334 72.533333c-84.650667 65.194667-139.264 167.509333-139.264 282.368v145.92c0 75.264 61.269333 136.533333 136.533333 136.533334h440.917333c75.264 0 136.533333-61.269333 136.533334-136.533334v-145.92c-0.085333-112.128-52.053333-212.309333-133.205334-277.76z m64.853334 423.765334c0 37.632-30.634667 68.266667-68.266667 68.266666H291.584c-37.632 0-68.266667-30.634667-68.266667-68.266666v-145.92c0-159.232 129.536-288.682667 288.682667-288.682667 159.232 0 288.682667 129.536 288.682667 288.682667v145.92z" fill="#333C4F" p-id="3593"></path><path d="M580.266667 794.88H443.733333c-18.773333 0-34.133333-15.36-34.133333-34.133333V759.466667c0-18.773333 15.36-34.133333 34.133333-34.133334h136.533334c18.773333 0 34.133333 15.36 34.133333 34.133334v1.28c0 18.773333-15.36 34.133333-34.133333 34.133333z" fill="#333C4F" p-id="3594"></path><path d="M553.642667 237.994667c-9.898667 0-19.2-4.266667-25.685334-11.690667l-26.112-29.866667-18.602666 26.88a34.2528 34.2528 0 0 1-28.074667 14.762667H395.093333c-18.858667 0-34.133333-15.274667-34.133333-34.133333s15.274667-34.133333 34.133333-34.133334h42.24l33.365334-48.298666c5.973333-8.704 15.616-14.08 26.197333-14.677334 10.581333-0.597333 20.736 3.754667 27.648 11.690667l44.629333 51.2 68.266667-0.426667h0.256a34.1248 34.1248 0 0 1 0.256 68.266667l-83.968 0.512c-0.170667-0.085333-0.256-0.085333-0.341333-0.085333z" fill="#333C4F" p-id="3595"></path></svg>'
    service_config={
        "url":"http://xx.xx:xx/xx",
        "headers":{
        },
        "data":{
        }
    }
    knowledge_config = '''
{
    "type": "str|api|file|sql",
    "content":"",
    "url":"",
    "headers": {},
    "data": {},
    "sql":'',
    "file":"/mnt/"
}
    '''

    add_form_extra_fields = {
        "name": StringField(
            label=_(datamodel.obj.lab('name')),
            description='英文名，小写',
            default='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
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
            default='text',
            widget=MySelect2Widget(),
            choices=[['text','文本对话'],['image',"图片"],  ['text_speech',"文本+语音"],['multi_modal', "多模态"], ['digital_human',"数字人"]],
            validators=[]
        ),
        "service_type": SelectField(
            label=_(datamodel.obj.lab('source_type')),
            description='推理服务类型',
            widget=Select2Widget(),
            default='chatgpt3.5',
            choices=[[x, x] for x in ["chatgpt3.5","chatgpt4", "chatglm", "微调模型",'AIGC','组合', '固定文案']],
            validators=[]
        ),
        "service_config": StringField(
            label=_(datamodel.obj.lab('service_config')),
            default=json.dumps(service_config,indent=4,ensure_ascii=False),
            description='服务配置',
            widget=MyBS3TextAreaFieldWidget(),
            validators=[DataRequired()]
        ),
        "knowledge": StringField(
            label=_(datamodel.obj.lab('knowledge')),
            default='',
            description='先验知识配置。如果先验字符串少于1800个字符，可以直接填写字符串，否则需要使用json配置',
            widget=MyBS3TextAreaFieldWidget(rows=5,tips=Markup('<pre><code>'+knowledge_config+"</code></pre>")),
            validators=[DataRequired()]
        ),
        "tips": StringField(
            label=_(datamodel.obj.lab('tips')),
            default='',
            description='提示输入，多个提示输入，多行配置',
            widget=MyBS3TextAreaFieldWidget(rows=3),
            validators=[]
        )
    }

    def pre_update(self, item):
        item.service_config = json.dumps(json.loads(item.service_config),indent=4,ensure_ascii=False)



    @action("copy", __("复制聊天"), confirmation=__('复制聊天窗配置'), icon="fa-copy",multiple=True, single=True)
    def copy(self, chats):
        if not isinstance(chats, list):
            chats = [chats]
        try:
            for chat in chats:
                new_chat = chat.clone()
                new_chat.created_on = datetime.datetime.now()
                new_chat.changed_on = datetime.datetime.now()
                db.session.add(new_chat)
                db.session.commit()
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            logging.error(e)
            raise e

        return redirect(request.referrer)


    edit_form_extra_fields = add_form_extra_fields
    all_history={}
    @expose('/chat/<chat_name>', methods=['POST','GET'])
    # @pysnooper.snoop()
    def chat(self,chat_name):
        args = request.json
        if not args:
            args = {}
        session_id = args.get('session_id','xxxxxx')
        request_id = args.get('request_id', str(datetime.datetime.now().timestamp()))
        search_text = args.get('search_text','你是谁？')
        search_audio = args.get('search_audio',None)
        search_image = args.get('search_image', None)
        stream = args.get('stream',False)
        # if stream=='true':
        #     stream = True
        begin_time = datetime.datetime.now()

        chat= db.session.query(Chat).filter_by(name=chat_name).first()
        if not chat:
            return jsonify({
                "status": 1,
                "message": '聊天不存在',
                "result": []
            })

        # 添加数据库记录
        if not stream:
            chatlog = ChatLog(
                username=str(g.user.username),
                chat_id=chat.id,
                query=search_text,
                answer="",
                manual_feedback="",
                answer_status="",
                answer_cost='0',
                err_msg="",
                created_on=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                changed_on=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            )
            db.session.add(chatlog)
            db.session.commit()
        else:
            chatlog = None

        return_result=[]
        return_message=''
        return_status=1
        finish = ''
        answer_status = 'making'
        err_msg = ''
        if session_id not in self.all_history:
            self.all_history[session_id]=[]
        history = self.all_history.get(session_id, [])
        history = [x for x in history if x]
        if chat.session_num:
            session_num = int(chat.session_num)
            if session_num>0:
                history=history[0-session_num:]

        if chat.service_type.lower()=='chatglm':
            return_status,text = self.chatglm(chat=chat,search_text=search_text,history=history)
            return_message = '失败' if return_status else "成功"
            return_result = [
                {
                    "text":text
                }
            ]
            # 追加记录的方法
            if search_text and text:
                self.all_history[session_id].append((search_text,text))

        if chat.service_type.lower()=='chatgpt4' or chat.service_type.lower()=='chatgpt3.5':
            if stream:
                res = self.chatgpt_stream(service_config=chat.service_config, knowledge=chat.knowledge,request_id=request_id, search_text=search_text, history=history)
                return res

            else:
                chatlog.answer_status = '准备发送chatgpt'
                db.session.commit()

                return_status, text = self.chatgpt(chat=chat, search_text=search_text, history=history)
                return_message = '失败' if return_status else "成功"
                answer_status = return_message
                return_result = [
                    {
                        "text": text
                    }
                ]
                # 追加记录的方法
                if search_text and text:
                    self.all_history[session_id].append((search_text, text))

        if chat.service_type == '固定文案':
            return_status,text = 0,"数洞心声+1，小星正在快马加鞭学习解决这类问题，我将保持与你联系~"
            return_message = '失败' if return_status else "成功"
            return_result = [ 
                {
                    "text": text
                }
            ]

        if chat.service_type.lower()=='aigc':
            return_status,image = self.aigc(chat=chat,search_text=search_text,history=history)
            return_message = '失败' if return_status else "成功"
            return_result = [
                {
                    "image":image
                }
            ]

        if chat.service_type.lower()=='组合':
            if chat.chat_type=='text_speech':
                # 发送的是文字，就返回文字
                if search_text:
                    return_status,text = self.chatglm(chat=chat,search_text=search_text,history=history)
                    return_message = 'chatglm失败' if return_status else "chatglm成功"
                    return_result = [
                        {
                            "text": text
                        }
                    ]
                    # 追加记录的方法
                    if search_text and text:
                        self.all_history[session_id].append((search_text, text))

                # 发送的是语音，就返回文字+语音
                elif search_audio:
                    return_status,text = self.asr(chat=chat,search_audio=search_audio,history=history)
                    return_message = 'asr失败' if return_status else "asr成功"
                    return_result = [
                        {
                            "text": text
                        }
                    ]
                    if not return_status:

                        return_status,text = self.chatglm(chat=chat, search_text=text, history=history)
                        return_message = 'chatglm失败' if return_status else "chatglm成功"
                        return_result = [
                            {
                                "text": text
                            }
                        ]
                        # 追加记录的方法
                        if search_text and text:
                            self.all_history[session_id].append((search_text, text))
                        if not return_status:
                            return_status,audio = self.tts(chat=chat, search_text=text, history=history)
                            return_message = 'tts失败' if return_status else "tts成功"
                            return_result = [
                                {
                                    "text": text,
                                    "audio": audio
                                }
                            ]


        # 添加数据库记录
        if chatlog:
            chatlog.query = search_text
            chatlog.answer = "\n".join(item.get('text','') for item in return_result)
            chatlog.answer_cost = str((datetime.datetime.now()-begin_time).total_seconds())
            chatlog.answer_status=answer_status,
            chatlog.err_msg = return_message
            db.session.commit()

        return jsonify({
            "status": return_status,
            "finish":finish,
            "message": return_message,
            "result": return_result
        })

    @expose('/chat/chatgpt/<chat_id>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def chatgpt_api(self, chat_id):
        chat = db.session.query(Chat).filter_by(id=int(chat_id)).first()
        search_text = request.json.get('search_text', '')
        search_audio = request.json.get('search_audio', None)
        status, text = self.chatglm(chat, search_text)
        return jsonify({
            "status": status,
            "message": '失败' if status else "成功",
            "result": [
                {
                    "text": text
                }
            ]
        })

    # @pysnooper.snoop()
    def chatgpt_stream(self, service_config,knowledge, request_id,search_text, history=[], stream=True):
        url = json.loads(service_config).get("chatgpt_url", '')
        headers = {
            'Accept': 'text/event-stream',
        }
        headers.update(json.loads(service_config).get("chatgpt_headers", {}))
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {"role": "system", 'content': knowledge},
                {'role': 'user', 'content': search_text},
            ],
            'temperature': 1,  # 问答发散度
            'top_p': 1,  # 同temperature
            'n': 1,  # top n可选值
            'stream': stream,
            'stop': 'elit proident sint',  #
            'max_tokens': 1500,  # 最大返回数
            'presence_penalty': 0,  #
            'frequency_penalty': 0,
            'user': 'user',
        }
        import sseclient
        data.update(json.loads(service_config).get("chatgpt_data", {}))
        request = requests.post(
            url,
            headers=headers,
            json=data,
            stream=stream
        )

        client = sseclient.SSEClient(request)

        def generate():
            back_message = ''
            for event in client.events():
                message = event.data
                if message != '[DONE]':
                    message = json.loads(event.data)['choices'][0].get('delta', {}).get('content', '')
                print(message)
                if message == '[DONE]':
                    break
                back_message = back_message + message
                back = "data:"+json.dumps(
                    {
                        "message":"success",
                        "status":0,
                        "result":[
                            {"text":back_message}
                        ]
                    },ensure_ascii=False
                )+"\n\n"
                yield back

        response = Response(stream_with_context(generate()), mimetype='text/event-stream')
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = 'keep-alive'
        return response

    # @pysnooper.snoop()
    def emb_knapsack(self,items, capacity):
        print(items)
        print(capacity)
        """
        Solve the knapsack problem using dynamic programming.
        """
        n = len(items)
        # Initialize a table to store the maximum value that can be obtained
        # for each combination of items and capacities.
        table = [[0 for j in range(capacity + 1)] for i in range(n + 1)]
        # Fill in the table row by row.
        for i in range(1, n + 1):
            item, weight, value = items[i - 1]
            for j in range(1, capacity + 1):
                # If the item is too heavy to include, skip it.
                if weight > j:
                    table[i][j] = table[i - 1][j]
                else:
                    # Consider two cases: including the item vs. not including the item.
                    included = value + table[i - 1][j - weight]
                    not_included = table[i - 1][j]
                    table[i][j] = max(included, not_included)
        # Trace back through the table to find the items that were included.
        result = []
        i = n
        j = capacity
        while i > 0 and j > 0:
            if table[i][j] != table[i - 1][j]:
                item, weight, value = items[i - 1]
                result.append(item)
                j -= weight
            i -= 1
        result.reverse()
        # Return the maximum value and the items that were included.
        return table[n][capacity], result

    # @pysnooper.snoop()
    def download_knowledge(self,knowledge):
        # 没有任何值就是空的
        if not knowledge:
            return ''
        try:
            knowledge_config = json.loads(knowledge)
        except Exception as e:
            print(e)
            return knowledge

        knowledge_type = knowledge_config.get("type", 'str')
        if knowledge_type == 'str':
            return knowledge_config.get("content", '')
        if knowledge_type == 'api':
            url = knowledge_config.get("url", '')
            headers = knowledge_config.get("headers", {})
            data = knowledge_config.get("data", {})
            res = requests.post(url, headers=headers, json=data)
            if res.status_code == 200:
                return res.text
        if knowledge_type == 'file':
            file_path = knowledge_config.get("file", '')
            if re.match('^/mnt', file_path):
                file_path = "/data/k8s/kubeflow/pipeline/workspace" + file_path.replace("/mnt", '')
            if os.path.exists(file_path):
                return open(file_path, mode='r').read()

        if knowledge_type == 'sql':
            return ''

    def emb_knowledge_split(self,knowledge, max_len = 200, mode=1):
        if mode==1:
            lst = knowledge.split('\n')
            spes = "。！？；，?.^"
            _lst = []
            for it in lst:
                while it:
                    #import time
                    #time.sleep(1)
                    #print(it)
                    if len(it) >max_len:
                        for sep in spes:
                            #time.sleep(1)
                            #print(sep)
                            if sep == '^':
                                tmp = it[:max_len]
                                _lst.append(tmp)
                                it = it[len(tmp):]
                                break
                            if sep in it[:max_len]:
                                tmp = sep.join(it[:max_len].split(sep)[:-1])
                                _lst.append(tmp + sep)
                                it = it[len(tmp)+1:]
                                break
                    else:
                        _lst.append(it)
                        it = ""
                    #print(_lst)
            return _lst
        else:
            return knowledge.split('\n')

    # @pysnooper.snoop()
    def emb_init(self,chat):
        global chatmodel
        if not chatmodel:
            chatmodel = ChatSimilarity(model_name_or_path="/text2vec-base-chinese")
        # 每一个小时更新一次
        if (str(chat.id) not in chatmodel.all_vector) or (datetime.datetime.now()-chatmodel.all_vector[str(chat.id)]['time']).total_seconds()>3600:
            # 从远端获取真实的文本语料
            knowledge = self.download_knowledge(chat.knowledge)
            # 添加语料进行embedding
            if knowledge:
                knowledge = self.emb_knowledge_split(knowledge)
                # knowledge = [x for x in knowledge.split('\n') if x.strip()]
                chatmodel.chat_add_corpus(chat_id=chat.id, corpus=knowledge)

    # @pysnooper.snoop()
    def topk(self,chat, query, topn=20, bag_size=3800):
        # 向量化，将预料embedding
        self.emb_init(chat)

        res = chatmodel.chat_most_similar(chat_id=chat.id,queries=[query], topn=topn)
        # print(res)
        q_id, c = list(res.items())[0]

        items = []
        for corpus_id, s in c.items():
            items.append((s,chatmodel.all_vector[str(chat.id)]['corpus'][corpus_id]))

        items = [(it[1], len(it[1]), 1./it[0]) for it in items]

        #knapsack()
        value, chosen_items = self.emb_knapsack(items, capacity = bag_size)  # 小于bag_size的将不会被加入进来，所以预料知识，每一行不能大于bag_size

        context = '\n'.join(chosen_items)
        return context

        # query = query + "。让我们一步一步思考，写下所有中间步骤，再产生最终解。"

    # @pysnooper.snoop()
    def chatgpt(self,chat,search_text,history=[]):
        try:
            if chat.knowledge:
                try:
                    json.loads(chat.knowledge)
                    knowledge = self.topk(chat, search_text)
                except Exception as e:
                    print(e)
                    knowledge = chat.knowledge  # 不是json就是原始语料
            else:
                knowledge=''

            url = json.loads(chat.service_config).get("chatgpt_url", '')
            headers={
                'Accept': 'application/json',
            }
            headers.update(json.loads(chat.service_config).get("chatgpt_headers", {}))

            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {"role": "system", 'content': knowledge},
                    {'role': 'user', 'content': search_text},
                ],
                'temperature': 1,   # 问答发散度
                'top_p': 1,   # 同temperature
                'n': 1,   # top n可选值
                'stream': False,
                'stop': 'elit proident sint',  #
                'max_tokens': 1500,   # 最大返回数
                'presence_penalty': 0,   #
                'frequency_penalty': 0,
                'user': 'user',
            }

            data.update(json.loads(chat.service_config).get("chatgpt_data", {}))
            res = requests.post(
                url,
                headers=headers,
                json=data
            )
            if res.status_code == 200 or res.status_code==201:
                # print(res.text)
                mes = res.json()['choices'][0]['message']['content']
                # print(mes)
                return 0,mes
            else:
                return 1, f'请求{url}失败'
        except Exception as e:
            return 1,"chatgpt报错："+str(e)


    @expose('/chat/chatglm/<chat_id>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def chatglm_api(self, chat_id):
        chat = db.session.query(Chat).filter_by(id=int(chat_id)).first()
        search_text = request.json.get('search_text', '')
        search_audio = request.json.get('search_audio', None)
        status, text = self.chatglm(chat, search_text)
        return jsonify({
            "status": status,
            "message": '失败' if status else "成功",
            "result": [
                {
                    "text": text
                }
            ]
        })


    # @pysnooper.snoop()
    def chatglm(self,chat,search_text,history=[]):
        try:
            query={
                "query":chat.knowledge+search_text
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


    @expose('/chat/tts/<chat_id>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def tts_api(self, chat_id):
        chat = db.session.query(Chat).filter_by(id=int(chat_id)).first()
        search_text = request.json.get('search_text', '')
        search_audio = request.json.get('search_audio', None)
        status, audio = self.tts(chat, search_text)
        return jsonify({
            "status": status,
            "message": '失败' if status else "成功",
            "result": [
                {
                    "audio": audio
                }
            ]
        })


    # @pysnooper.snoop()
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


    @expose('/chat/asr/<chat_id>', methods=['POST', 'GET'])
    # @pysnooper.snoop()
    def asr_api(self, chat_id):
        chat = db.session.query(Chat).filter_by(id=int(chat_id)).first()
        search_text = request.json.get('search_text', '')
        search_audio = request.json.get('search_audio', None)
        status, text = self.asr(chat, search_audio)
        return jsonify({
            "status": status,
            "message": '失败' if status else "成功",
            "result": [
                {
                    "text": text
                }
            ]
        })

    # @pysnooper.snoop()
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

    @expose('/chat/aigc/<chat_id>', methods=['POST','GET'])
    # @pysnooper.snoop()
    def aigc_api(self,chat_id):
        chat = db.session.query(Chat).filter_by(id=int(chat_id)).first()
        search_text = request.json.get('search_text', '')
        search_audio = request.json.get('search_audio', None)
        status,image = self.aigc(chat,search_text)
        return jsonify({
            "status": status,
            "message": '失败' if status else "成功",
            "result": [
                {
                    "image": image
                }
            ]
        })

    # @pysnooper.snoop()
    def aigc(self,chat,search_text,history=[]):
        try:
            url = json.loads(chat.service_config).get("aigc_url",'')
            headers = json.loads(chat.service_config).get("aigc_headers",{})
            data = {
                "text": chat.knowledge + search_text
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


    # info接口响应修正
    # @pysnooper.snoop()
    def pre_list_res(self,_response):
        for chat in _response['data']:
            chat['tips']=[x for x in chat['tips'].split('\n') if x] if chat['tips'] else []
        return _response

appbuilder.add_api(Chat_View)



