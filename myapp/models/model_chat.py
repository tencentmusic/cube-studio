import json

from flask_appbuilder import Model

from sqlalchemy import Text

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

    name = Column(String(200), nullable=True, default='')
    icon = Column(Text, nullable=True, default='')
    label = Column(String(200), nullable=True, default='')
    doc = Column(String(200), nullable=True, default='')
    session_num = Column(String(200), nullable=True, default='0')  # 会话保持的个数，默认为1，只记录当前的会话
    chat_type = Column(String(200), nullable=True, default='text')  # 聊天会话的界面类型 # text文本对话，speech 语音对话，text_speech文本加语音对话，multi_modal多模态对话，digital_human数字人
    hello = Column(String(200), nullable=True, default='')
    pre_question = Column(Text, nullable=True, default='')   # 加载问题前面的先验知识
    service_type = Column(String(200), nullable=True, default='')  # 推理服务的类型   [chatgpt,gptglm]
    service_url = Column(String(2000), nullable=True, default='')  # 推理服务的地址
    owner = Column(String(2000), nullable=True, default='*')  # 可访问用户，* 表示所有用户

    expand = Column(Text, nullable=True, default='{}')





