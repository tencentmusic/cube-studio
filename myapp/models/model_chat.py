import json

from flask_appbuilder import Model

from sqlalchemy import Text
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from myapp import app
from myapp.models.helpers import ImportMixin
from sqlalchemy import Column, Integer, String

from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config


class Chat(Model,MyappModelBase):
    __tablename__ = 'chat'

    id = Column(Integer, primary_key=True)

    name = Column(String(200), nullable=True, default='',unique=True)
    icon = Column(Text, nullable=True, default='')
    label = Column(String(200), nullable=True, default='')
    doc = Column(String(200), nullable=True, default='')
    session_num = Column(String(200), nullable=True, default='0')  # 会话保持的个数，默认为1，只记录当前的会话
    chat_type = Column(String(200), nullable=True, default='text')  # 聊天会话的界面类型 # text文本对话，speech 语音对话，text_speech文本加语音对话，multi_modal多模态对话，digital_human数字人
    hello = Column(String(200), nullable=True, default='')   # 欢迎语
    tips = Column(String(4000), nullable=True, default='')   #  提示词数组  \n分割
    knowledge = Column(Text, nullable=True, default='')   # 加载问题前面的先验知识
    service_type = Column(String(200), nullable=True, default='')  # 推理服务的类型   [chatgpt,gptglm]
    service_config = Column(Text, nullable=True, default='{}')  # 推理服务的配置，url  header  json等
    owner = Column(String(2000), nullable=True, default='*')  # 可访问用户，* 表示所有用户

    expand = Column(Text, nullable=True, default='{}')


    def clone(self):
        return Chat(
            name=self.name,
            icon=self.icon,
            label=self.label+"(副本)",
            doc=self.doc,
            session_num=self.session_num,
            chat_type=self.chat_type,
            hello=self.hello,
            knowledge=self.knowledge,
            service_type=self.service_type,
            service_config=self.service_config,
            owner=self.owner,
            expand=self.expand
        )

class ChatLog(Model,MyappModelBase):
    __tablename__ = 'chat_log'
    id = Column(Integer, primary_key=True)
    username = Column(String(400), nullable=False, default='')
    chat_id = Column(Integer)
    query = Column(String(5000), nullable=False, default='')
    answer = Column(String(5000), nullable=False, default='')
    manual_feedback = Column(String(400), nullable=False, default='')
    answer_status = Column(String(400), nullable=False, default='')
    answer_cost = Column(String(400), nullable=False, default='')
    err_msg = Column(String(5000), nullable=False, default='')
    created_on = Column(DateTime, nullable=True, default='')
    changed_on = Column(DateTime, nullable=True, default='')



