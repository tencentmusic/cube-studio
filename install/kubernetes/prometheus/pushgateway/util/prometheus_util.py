# coding=utf-8
import base64
import asyncio
import json,datetime,time
import logging
import os
import io
import prometheus_client
from prometheus_client import Counter,Gauge
from prometheus_client.core import CollectorRegistry
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class Promethus():

    def __init__(self):
        self.loop = asyncio.get_event_loop()  # 获取全局轮训器
        self.registry = CollectorRegistry()   # 存放所有Metrics的容器，以Name-Metric（Key-Value）形式维护其中的Metric对象。
        self.all_metric={}

    # 可以包含多个metric,以字典的形式传输
    async def label_data(self,json_data):
        # print(json_data)
        try:
            for metric_name in json_data:

                metric_post=json_data[metric_name]
                print(metric_post)
                labels=metric_post.get('labels',[])
                describe=metric_post.get('describe','')

                # update 覆盖原有值
                # clear 删除
                # keep 保留原样
                # add 值添加
                # reset 值设置为0
                exist_not_update_type= metric_post.get('exist_not_update_type',None)
                exist_update_type = metric_post.get('exist_update_type', None)
                not_exist_update_type = metric_post.get('not_exist_update_type', None)
                pull_finish_deal_type = metric_post.get('pull_finish_deal_type', None)   # 被拉取以后的处理行为
                # 创建metric
                if metric_name not in self.all_metric:
                    if labels:
                        await self.create_metric(metric_name,labels,describe)     # labels 是不能变的.只不过每种labels取值时的metric_value 是否要保留是不一定了.
                    else:
                        continue
                # 更新metric属性
                if exist_not_update_type:
                    self.all_metric[metric_name]['exist_not_update_type']=exist_not_update_type
                if exist_update_type:
                    self.all_metric[metric_name]['exist_update_type']=exist_update_type
                if not_exist_update_type:
                    self.all_metric[metric_name]['not_exist_update_type']=not_exist_update_type
                if pull_finish_deal_type:
                    self.all_metric[metric_name]['pull_finish_deal_type']=pull_finish_deal_type


                exist_not_update_type = self.all_metric[metric_name]['exist_not_update_type']
                exist_update_type = self.all_metric[metric_name]['exist_update_type']
                not_exist_update_type = self.all_metric[metric_name]['not_exist_update_type']


                # 对数据做一下变形
                data_tuple={}
                if 'data' in metric_post:
                    for one_data in metric_post['data']:
                        attr,value=one_data
                        attr=tuple(attr)
                        data_tuple[attr]=value
                # print(data_tuple)
                # 按规则更新数据
                # 对已存在但是没有更新的数据的处理,默认不变化
                if exist_not_update_type=='clear':
                    for attr in self.all_metric[metric_name]['data']:
                        if attr not in data_tuple:
                            del self.all_metric[metric_name]['data']
                elif exist_not_update_type=='reset':
                    for attr in self.all_metric[metric_name]['data']:
                        if attr not in data_tuple:
                            self.all_metric[metric_name]['data']=0

                # 对已存在,同时也更新的数据进行处理,默认不变化
                if exist_update_type == 'update':
                    for attr in data_tuple:
                        if attr in self.all_metric[metric_name]['data']:
                            self.all_metric[metric_name]['data'][attr] = data_tuple[attr]
                elif exist_update_type == 'add':
                    for attr in data_tuple:
                        if attr in self.all_metric[metric_name]['data']:
                            self.all_metric[metric_name]['data'][attr] += data_tuple[attr]
                elif exist_update_type == 'clear':
                    for attr in data_tuple:
                        if attr in self.all_metric[metric_name]['data']:
                            del self.all_metric[metric_name]['data'][attr]
                elif exist_update_type == 'reset':
                    for attr in data_tuple:
                        if attr in self.all_metric[metric_name]['data']:
                            self.all_metric[metric_name]['data'][attr]=0

                # 对不存在,同时更新的数据进行处理.默认不变化
                if not_exist_update_type == 'reset':
                    for attr in data_tuple:
                        if attr not in self.all_metric[metric_name]['data']:
                            self.all_metric[metric_name]['data'][attr] = 0
                elif not_exist_update_type == 'add' or not_exist_update_type == 'update' :
                    for attr in data_tuple:
                        if attr not in self.all_metric[metric_name]['data']:
                            self.all_metric[metric_name]['data'][attr] = data_tuple[attr]

                return True

        except Exception as e:
            print(e)
            return False
        return False


    # 删除matric
    async def delete(self,metric_name):
        del self.all_metric[metric_name]

    # 读取metric的数据
    async def get_metric_prometheus(self,metric_name):
        if metric_name in self.all_metric[metric_name]:
            # 根据data生成prometheus格式数据,
            metric = Gauge(metric_name, self.all_metric[metric_name]['describe'], self.all_metric[metric_name]['labels'])

            for attr in self.all_metric[metric_name]['data']:
                metric.labels(*attr).set(self.all_metric[metric_name]['data'][attr])
            prometheus_data = prometheus_client.generate_latest(metric)

            # 处理拉取数据后逻辑
            if self.all_metric[metric_name]['pull_finish_deal_type']=='clear':
                self.all_metric[metric_name]['data']={}
            elif self.all_metric[metric_name]['pull_finish_deal_type']=='reset':
                for attr in self.all_metric[metric_name]['data']:
                    self.all_metric[metric_name]['data'][attr]=0

            return prometheus_data

        return None

    # 获取所有的metrics数据
    async def get_metrics_prometheus(self,onlyread=False):
        self.registry= CollectorRegistry()
        for metric_name in self.all_metric:
            # Gauge默认存放在全局register中,且全局register中不能存在相同名称的metric
            metric = Gauge(name=metric_name, documentation=self.all_metric[metric_name]['describe'],labelnames=self.all_metric[metric_name]['labels'],registry=self.registry)

            for attr in self.all_metric[metric_name]['data']:
                metric.labels(*attr).set(self.all_metric[metric_name]['data'][attr])

            if not onlyread:
                # 处理拉取数据后逻辑
                if self.all_metric[metric_name]['pull_finish_deal_type']=='clear':
                    self.all_metric[metric_name]['data']={}
                elif self.all_metric[metric_name]['pull_finish_deal_type']=='reset':
                    for attr in self.all_metric[metric_name]['data']:
                        self.all_metric[metric_name]['data'][attr]=0

        return prometheus_client.generate_latest(self.registry)

    #读取metric的信息和数据
    async def get_metric(self,metric_name):
        return self.all_metric[metric_name]

    # 获取所有metric信息
    async def get_metrics(self):
        return self.all_metric

    # labels为可以为该数据打的标签
    async def create_metric(self,metric_name,labels,describe=''):
        print('create metric %s'%metric_name)

        if metric_name in self.all_metric:
            del self.all_metric[metric_name]
        metric={
            'labels':labels,
            'describe':describe,
            'data':{},
        }
        self.all_metric[metric_name] = metric






