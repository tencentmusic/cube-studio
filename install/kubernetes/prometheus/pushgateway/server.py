# pip install webhooks
import sys
import os

dir_common = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dir_common)   # 将根目录添加到系统目录,才能正常引用common文件夹
import logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import argparse
from aiohttp import web
import aiohttp
import copy
import asyncio
import base64
import logging
import time,datetime
import json
import requests
from aiohttp.web import middleware
import aiohttp_cors   # 支持跨域请求
from util.prometheus_util import *
from util.config import *
import prometheus_client
from prometheus_client import Counter,Gauge
from prometheus_client.core import CollectorRegistry
from prometheus_client import CollectorRegistry, Gauge
import socket
import requests
from webhooks import webhook
from webhooks.senders import targeted
loop = asyncio.get_event_loop()  # 获取全局轮训器
hostName = socket.gethostname()
routes = web.RouteTableDef()
promethus=Promethus()

Sender = os.getenv('sender', 'TME_DataInfra')
Receiver = os.getenv('receiver', '')
Sender_type = os.getenv('Sender_type', 'wechat')

def push_message(sender_type,**args):
    if sender_type=='wechat':
        push_wechat(args['message'],args['sender'],args['receiver'])
    if sender_type=='username_group':
        push_username_group(args['message'],args['sender'])


# 微信公共号告警，指向个人推送
def push_wechat(message,sender,receiver):
    if not sender or not receiver or not message:
        logging.info('no sender %s, or not receiver %s, or no message %s '%(sender,receiver,message))
        return
    if type(receiver)==str:
        receiver = receiver.split(",")
        receiver = [str(x).strip() for x in receiver if x]
    data = {
        "Sender": sender,
        "Rcptto": receiver,
        "isText": message
    }
    jsondata = json.dumps(data)
    logging.info('begin to send wechat %s' % jsondata)
    import urllib
    import urllib.parse
    import urllib.request
    values = urllib.parse.urlencode({"data": jsondata})
    resp = urllib.request.urlopen('http://api.weixin.oa.com/itilalarmcgi/sendmsg?%s'%values, timeout=10)
    logging.info('reveive resp from wechat: %s'% resp.read().decode("unicode_escape"))

# 企业微信群推送，sender为企业微信群的的key,message为字典数据
def push_username_group(message,sender):
    data = {
        "msgtype": "text",
        "text": {
            "content": message
        }
    }
    url = 'http://in.qyapi.weixin.qq.com/cgi-bin/webhook/send?key=%s'%sender
    logging.info('begin to send username group %s'%url)
    resp = requests.post(url,timeout=10,json=data)
    logging.info('reveive resp from username: %s'%resp.content)


# 推送数据
@routes.post('/metrics')
async def post_data(request):   # 异步监听，只要一有握手就开始触发
    try:
        data = await request.json()    # 等待post数据完成接收，只有接收完成才能进行后续操作.data['key']获取参数
    except Exception as e:
        logging.error("image file too large or cannot convert to json")
        return web.json_response(write_response(ERROR_FILE_LARGE,"image file too large or cannot convert to json",{}))

    logging.info('receive metrics data %s' % datetime.datetime.now())
    status = await promethus.label_data(data['metrics'])     # 包含记录信息，处理图片，存储图片，token过期以后要请求license服务器

    logging.info('save metrics data finish %s, %s' % (datetime.datetime.now(),str(status)))
    header = {"Access-Control-Allow-Origin": "*", 'Access-Control-Allow-Methods': 'GET,POST'}
    if status:
        return web.json_response(write_response(0,"success",{}),headers=header)
    else:
        return web.json_response(write_response(1, "error", {}), headers=header)


# 推送数据
@routes.post('/{client}/webhook')
async def client_webhook(request):   # 异步监听，只要一有握手就开始触发
    logging.info('=================begin a webhook')
    global Receiver,Sender,Sender_type
    client = request.match_info['client']
    try:
        data = await request.json()    # data可能是dict或者list
        args = dict(request.query)
        logging.info('src data:%s'%data)
        logging.info('src args:%s'%args)
        if type(data) == list:
            data = data[0]
        if args:
            for key in args:
                data[key] = args[key]
        for key in list(data.keys()):
            if key.lower() == 'username':
                Receiver = data[key]
                del data[key]
            if key.lower() == 'sender_type':
                Sender_type = data[key]
                del data[key]
            if key.lower() == 'sender':
                Sender = data[key]
                del data[key]
        if 'message' in data:
            message = data['message']
        else:
            message = json.dumps(data, indent=4, ensure_ascii=False).encode('utf8').decode('utf8')

        try:
            if client=='grafana':
                data={
                    'title': data['title'],
                    'message':data['message'],
                    'state':data['state'],
                }
                message = json.dumps(data,indent=4, ensure_ascii=False).encode('utf8').decode('utf8')

            if client=='rancher':
                message = json.dumps(data['labels'],indent=4)
            if client=='alertmanager':
                data_label = copy.deepcopy(data['alerts'][0]['labels'])
                if 'job' in data_label: del data_label['job']
                if 'service' in data_label: del data_label['service']
                if 'prometheus' in data_label: del data_label['prometheus']
                if 'endpoint' in data_label: del data_label['endpoint']
                if 'pod' in data_label: del data_label['pod']
                if 'instance' in data_label: del data_label['instance']
                data_push={
                    'labels':data_label,
                    # 'annotations':data['alerts'][0]['annotations'],
                    'status':data['alerts'][0]['status'],
                }
                message = json.dumps(data_push,indent=4, ensure_ascii=False).encode('utf8').decode('utf8')
            if client=='superset':
                message = data['message'] # , ensure_ascii=False,indent=4)  # .encode('utf8').decode('utf8')
        except Exception as e:
            logging.error(e)
        logging.info('%s %s %s %s'%(Sender_type,Sender,Receiver,message))
        push_message(sender_type=Sender_type,message=message+"\n from %s"%client,sender=Sender,receiver=Receiver)

    except Exception as e:
        logging.info(e)
        return web.json_response(write_response(ERROR_FILE_LARGE,"can not access json",{}))

    logging.info('finish deal webhook data %s' % datetime.datetime.now())

    return web.json_response(write_response(0,"success",{}))

# 推送数据
@routes.post('/junopodstop/customwebhook')
async def client_webhook(request):   # 异步监听，只要一有握手就开始触发
    logging.info('=================begin a customwebhook')
    global Receiver, Sender, Sender_type
    client = request.match_info['client']
    try:
        data = await request.json()    # data可能是dict或者list
        args = dict(request.query)
        logging.info('src data:%s'%data)
        logging.info('src args:%s'%args)
        username = data['username']
        try:
            message = data['message']
        except Exception as e:
            logging.info(e)
        logging.info('%s %s %s %s'%(Sender_type,Sender,username,message))
        push_message(sender_type=Sender_type,message=message+"\n from %s"%client,sender=Sender,receiver=username)

    except Exception as e:
        logging.info(e)
        return web.json_response(write_response(ERROR_FILE_LARGE,"can not access json",{}))

    logging.info('finish deal webhook data %s' % datetime.datetime.now())

    return web.json_response(write_response(0,"success",{}))



# 读取数据
@routes.get('/')
async def default(request):
    return web.Response(text="OK")


@routes.get('/metrics')
async def get_data(request):
    data = await promethus.get_metrics_prometheus()
    return web.Response(body=data, content_type="text/plain")  # 将计数器的值返回


if __name__ == '__main__':
    # init_logger(module_name="face_det")   # 初始化日志配置
    init_console_logger()

    app = web.Application(client_max_size=int(LOCAL_SERVER_SIZE)*1024**2,debug=True)    # 创建app，设置最大接收图片大小为2M
    app.add_routes(routes)     # 添加路由映射
    # 编写支持跨域的路由
    core = aiohttp_cors.setup(app,defaults={
        '*':aiohttp_cors.ResourceOptions(
            allow_methods='*',
            allow_credentials=True,
            allow_headers='*',
            expose_headers='*'
        )
    })
    for route in list(app.router.routes()):
        core.add(route)

    logging.info('server start,port is %s, datetime is %s'%(str(LOCAL_SERVER_PORT),str(datetime.datetime.now())))
    web.run_app(app,host=LOCAL_SERVER_IP,port=LOCAL_SERVER_PORT)   # 启动app
    logging.info('server close：%s'% datetime.datetime.now())



