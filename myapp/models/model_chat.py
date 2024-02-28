import json
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder import Model
from sqlalchemy import Text
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from myapp import app
from sqlalchemy import Column, Integer, String

from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config


class Chat(Model,MyappModelBase):
    __tablename__ = 'chat'

    id = Column(Integer, primary_key=True,comment='id主键')

    name = Column(String(200), nullable=True, default='',unique=True,comment='英文名')
    icon = Column(Text, nullable=True, default='',comment='图标svg内容')
    label = Column(String(200), nullable=True, default='',comment='中文名')
    doc = Column(String(200), nullable=True, default='',comment='文档链接')
    session_num = Column(String(200), nullable=True, default='0',comment='会话保持的个数，默认为1，只记录当前的会话')  #
    chat_type = Column(String(200), nullable=True, default='text',comment='聊天会话的界面类型 # text文本对话，speech 语音对话，text_speech文本加语音对话，multi_modal多模态对话，digital_human数字人')  #
    hello = Column(String(200), nullable=True, default='',comment='欢迎语')
    tips = Column(String(4000), nullable=True, default='',comment='提示词数组')
    knowledge = Column(Text, nullable=True, default='',comment='加载问题前面的先验知识')   #
    prompt = Column(Text, nullable=True, default='{{prompt}}',comment='提示词模板')  #
    service_type = Column(String(200), nullable=True, default='',comment=' 推理服务的类型   [chatgpt,gptglm]')  #
    service_config = Column(Text, nullable=True, default='{}',comment='推理服务的配置，url  header  json等')  #
    owner = Column(String(2000), nullable=True, default='*',comment='可访问用户，* 表示所有用户')  #

    expand = Column(Text, nullable=True, default='{}',comment='扩展参数')


    def clone(self):
        return Chat(
            name=self.name,
            icon=self.icon,
            label=self.label+__("(副本)"),
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
    id = Column(Integer, primary_key=True,comment='id主键')
    username = Column(String(400), nullable=False, default='',comment='用户名')
    chat_id = Column(Integer,comment='场景id')
    query = Column(String(5000), nullable=False, default='',comment='问题')
    answer = Column(String(5000), nullable=False, default='',comment='回答')
    manual_feedback = Column(String(400), nullable=False, default='',comment='反馈')
    answer_status = Column(String(400), nullable=False, default='',comment='回答状态')
    answer_cost = Column(String(400), nullable=False, default='',comment='话费时长')
    err_msg = Column(Text,comment='报错消息')
    created_on = Column(DateTime, nullable=True, default='',comment='创建时间')
    changed_on = Column(DateTime, nullable=True, default='',comment='修改时间')



