import logging
import os
import requests
import pysnooper
import json
import re,datetime,time,os,sys,io
import inspect
from cubestudio.request.model_client import Client,HOST,USERNAME
from inspect import isfunction
import socketio
from cubestudio.utils.py_inspect import get_classes
MODEL_CLASS = {
}

# 根据名称获取model类
def get_model_class_client(model_name):
    global MODEL_CLASS
    if not MODEL_CLASS:
        all_class = get_classes(sys.modules[__name__])
        for class_name in all_class:
            if issubclass(all_class[class_name],Model):
                MODEL_CLASS[class_name]=all_class[class_name]

    model_class = MODEL_CLASS.get(model_name,None)
    return model_class

# 没有client，没法进行初始化，在没有client前的Model，必须先传入client
class Model():
    # 类属性，一个类，设置以后，就不再变化，
    client=None
    # @pysnooper.snoop()
    def __init__(self,**kwargs):
        # print(kwargs)
        self.client=Client(self.__class__)
        for key in kwargs:
            col_type = self.client.columns_info.get(key, {}).get("type", '')
            if col_type=='Relationship':  # 如果是个外键字段，就翻译为外键mode类
                model_class_name = self.client.columns_info.get(key, {}).get("relationship",'')
                if not kwargs[key]:   # 外键对象为空
                    setattr(self, key, None)
                else:
                    model_class = get_model_class_client(model_class_name)
                    if model_class and type(kwargs[key])==dict:
                        # print(self.__class__,key,kwargs[key])
                        setattr(self, key, model_class(**kwargs[key]))
                    else:
                        setattr(self, key, kwargs[key])
            else:
                setattr(self,key,kwargs[key])

        # 如果是对象，就翻译为对象

    # @pysnooper.snoop(watch_explode=("back",))
    def to_dict(self):
        back={}
        show_columns = getattr(self.client,'show_columns',[])
        # print(show_columns)
        for col in show_columns:

            valus=getattr(self,col,None)
            # print(col,valus,type(valus))
            # 函数不正则
            if hasattr(valus, '__call__'):
                continue

            # model类型属性，调用其正则化
            if isinstance(valus,Model):
                back[col]=valus.to_dict()
                # print(col,valus.to_dict())
            else:
                back[col]=valus
        return back

    def update(self,**kwargs):
        return self.client.update(self,**kwargs)

    def delete(self):
        self.client.delete(self.id)

    # def __str__(self):
    #     return json.dumps(self.to_dict(),ensure_ascii=False,indent=4)
