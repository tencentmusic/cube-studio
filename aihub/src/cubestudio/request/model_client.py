import os
import requests
import pysnooper
import json
import re,datetime,time,os,sys,io
import types

USERNAME=''
TOKEN=''
HOST=''
def init(host,username,token):
    global HOST,USERNAME,TOKEN
    try:
        # print('\n\n\n')
        req_path=host.rstrip('/')+"/users/api/"
        headers={
            'Content-Type': 'application/json',
            "Authorization":token
        }
        data={
            "filters": [
                {
                    "col": "username",
                    "opr": "eq",
                    "value": username
                }
            ]
        }
        res = requests.request("GET",req_path,timeout=30,headers=headers,json=data,allow_redirects=False)
        # print(res.content)
        if res.status_code==200:
            result = res.json()
            user = result.get("result",{}).get("data",[])[0]
            exist_username = user.get("username",'')
            if exist_username==username:
                USERNAME=username
                HOST=host.rstrip('/')
                TOKEN=token
                print('初始化验证成功')
            else:
                raise Exception("token和用户名不匹配")
        else:
            print(res.content)
            raise Exception("请求服务端验证失败")
    except Exception as e:
        raise Exception("token和用户名不匹配")


# 不能频繁进行实例化
class Model_Client():

    path=''
    model_class = None

    related={}
    route_base=''
    primary_key='id'
    action={}
    help_url=''
    columns_info={}
    label_title=edit_title=add_title=show_title=list_title=''
    show_fieldsets=edit_fieldsets=add_fieldsets=[]
    list_columns=order_columns=show_columns=edit_columns=add_columns=[]
    description_columns=label_columns={}
    filters={}
    user_permissions={}
    permissions=[]
    import_data=False
    download_data=False
    pagesize=100
    info={}

    def __init__(self,model_class):
        self.model_class = model_class  # 绑定model类
        # 将model中的信息实例化到当前
        self.path=model_class.path


        # 全局信息
        self.host=HOST
        self.token=TOKEN
        self.username=USERNAME

        # 查询获取client信息
        self._info()
        if self.info:
            for key in ['related','route_base','primary_key','action','help_url','columns_info',
                        'label_title','edit_title','add_title','show_title','list_title',
                        'show_fieldsets','edit_fieldsets','add_fieldsets',
                        'list_columns','order_columns','description_columns','label_columns','show_columns','edit_columns','add_columns',
                        'filters','user_permissions','permissions','import_data','download_data','page_size']:
                setattr(self,key,self.info[key])
        else:
            print('链接失败')

    # @pysnooper.snoop()
    def req(self,path,method="GET",data=None):
        # print(path,method,json)
        # print('\n\n\n')
        req_path=self.host.rstrip('/')+path
        headers={
            'Content-Type': 'application/json',
        }
        if self.token:
            headers['Authorization']=self.token
        res = requests.request(method.upper(),req_path,json=data,timeout=30,headers=headers,allow_redirects=False)
        if res.status_code==200:
            # print(res.json())
            try:
                result = res.json()
            except Exception as e:
                result=res.content.decode('utf-8')
            return result
        elif res.status_code==302 or res.status_code==301:
            return res.headers.get('location','')
        else:
            # print(path,method,data)
            raise Exception(res.content)


    # @pysnooper.snoop()
    def _info(self):
        self.info = self.req(self.path + "/_info")

    # 把默认值先加上
    # @pysnooper.snoop(watch_explode=["kwargs"])
    def add(self,**kwargs):
        if 'can_add' not in self.permissions:
            print(self.__class__.__name__,"对象不允许add")
            return
        # 校验必填字段是都没填
        for col in self.add_columns:
            col_name = col['name']
            if col_name not in kwargs:
                kwargs[col_name]=col['default']
            # print(col, kwargs[col_name])
            validators = col['validators']
            for val in validators:
                val_type = val['type']
                val_regex = val.get('regex','')
                val_min = val.get('min',0) if val.get('min',0) else 0
                val_max = val.get('max',10000000) if val.get('max',10000000) else 10000000
                if val['type']=='DataRequired' and kwargs.get(col_name,None)==None:
                    raise Exception(f'字段{col_name}为必填')
                if val['type']=='Regexp' and col_name in kwargs and not re.search(val_regex, kwargs.get(col_name,'')):
                    raise Exception(f'字段{col_name}必须满足正则表达式{val["regex"]}')
                if val['type']=='Length' and col_name in kwargs and (len(kwargs.get(col_name,''))<val_min or len(kwargs.get(col_name,''))>val_max):
                    raise Exception(f'字段{col_name}长度必须在{val["min"]}~{val["max"]}')

        # 只保留允许填写的部分
        data = {}
        # print(self.add_columns)
        for key in kwargs:
            columns = [col['name'] for col in self.add_columns]
            if key in columns:
                from cubestudio.request.model import Model
                data[key]=getattr(kwargs[key],self.primary_key) if issubclass(kwargs[key].__class__,Model) else kwargs[key]
            else:
                print(key,"为添加时多余字段")
        item = self.req(self.path + "/",method='POST',data=data)
        # print(item)
        if item and item.get("status",1)==0:
            id = item.get("result",{}).get(self.primary_key,0)
            if id:
                model = self.show(id)
                return model



    # 更新
    # @pysnooper.snoop()
    def update(self,model,**kwargs):
        if 'can_edit' not in self.permissions:
            print(self.__class__.__name__,"对象不允许update")
            return
        # 校验必填字段是都没填
        for col in self.edit_columns:
            col_name = col['name']
            if col_name in kwargs:
                validators = col['validators']
                for val in validators:
                    val_type = val['type']
                    val_regex = val.get('regex', '')
                    val_min = val.get('min', 0) if val.get('min', 0) else 0
                    val_max = val.get('max', 10000000) if val.get('max', 10000000) else 10000000

                    if val['type'] == 'DataRequired' and not kwargs.get(col_name, None):
                        raise Exception(f'字段{col_name}为必填')
                    if val['type'] == 'Regexp' and not re.search(val_regex, kwargs.get(col_name, '')):
                        raise Exception(f'字段{col_name}必须满足正则表达式{val["regex"]}')
                    if val['type'] == 'Length' and (len(kwargs.get(col_name, '')) < val_min or len(kwargs.get(col_name, '')) > val_max):
                        raise Exception(f'字段{col_name}长度必须在{val["min"]}~{val["max"]}')

        # 只保留允许填写的部分
        data = {}
        # print(self.add_columns)
        for key in kwargs:
            columns = [col['name'] for col in self.edit_columns]
            if key in columns:
                from cubestudio.request.model import Model
                data[key]=getattr(kwargs[key],self.primary_key) if issubclass(kwargs[key].__class__,Model) else kwargs[key]
            else:
                print(key, "为修改时多余字段")
        #
        item = self.req(self.path + "/"+str(getattr(model,self.primary_key,'')), method='PUT', data=data)

        if item and item.get("status",1)==0:
            id = item.get("result", {}).get(self.primary_key, 0)
            if id:
                model = self.show(id)
                return model

    def show(self,id):
        if 'can_show' not in self.permissions:
            print(self.__class__.__name__,"对象不允许show")
            return

        item = self.req(self.path + "/"+str(id))
        if item and item.get("status",1)==0:
            # print(item)
            model = self.model_class(**item.get("result",{}))
            # print(model)
            return model

    def one(self,**kwargs):
        items = self.list(search=kwargs)
        if items:
            return items[0]
        else:
            return None

    def search(self,**kwargs):
        items = self.list(search=kwargs)
        if items:
            return items
        else:
            return None

    # @pysnooper.snoop()
    def list(self,search=None,order_column=None,order_direction='desc'):
        if 'can_list' not in self.permissions:
            print(self.__class__.__name__,"对象不允许list")
            return
        data={
            "page":0,
            "pagesize": self.pagesize,
            "order_column":order_column,
            "order_direction":order_direction,
            "columns":list(set(getattr(self,'show_columns',[])+getattr(self,'list_columns',[])))
        }
        if search and type(search)==dict:
            data['filters']=[]
            for key in search:
                data['filters'].append(
                    {
                        "col": key,
                        "opr": "eq",
                        "value": search[key]
                    }
                )
        result = self.req(self.path + "/",data=data)
        if result:
            result = result.get("result", {}).get("data",[])
            # print(result)
            model_list=[]
            for item in result:
                # print(item)
                model = self.model_class(**item)
                model_list.append(model)

            return model_list


    # 添加或者更新
    # @pysnooper.snoop()
    def add_or_update(self,**kwargs):
        model=None
        if len(kwargs.keys()):
            # 第一个参数为过滤查询使用
            search_name = list(kwargs.keys())[0]
            search_value = list(kwargs.values())[0]
            data={
                search_name:search_value
            }
            model = self.one(**data)
        if not model:
            return self.add(**kwargs)
        else:
            return self.update(model,**kwargs)

    def delete(self,id):
        if 'can_delete' not in self.permissions:
            print(self.__class__.__name__,"对象不允许delete")
            return
        item = self.req(self.path + "/" + id, method='DELETE')
        return item

    def single_action(self,action,id):
        item = self.req(self.path + f"/action/{action}/{id}")
        return item

    def multi_action(self,action,ids):
        items = self.req(self.path + f"/multi_action/{action}",method="POST",data={"ids":ids})
        return items




all_client={}
def Client(model_class):
    if type(model_class) != type:
        print('输入为实例,现转化为class')
        model_class = model_class.__class__
    if model_class.__name__ not in all_client:
        all_client[model_class.__name__]=Model_Client(model_class)
    return all_client[model_class.__name__]

