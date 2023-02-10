import datetime
import random
import time
import flask_socketio
from myapp import app,conf,socketio
import pysnooper

@socketio.on("server_event_name",namespace='/message')
def server_event_name(message,**kwargs):
    print(message)
    print(type(message))
    print(kwargs)
    print("server has connected")
    msg = '示例消息'
    index = 1
    while index<10:  # 循环监听
        # 监听链接,接收数据
        print(msg)
        message = msg + str(datetime.datetime.now())
        flask_socketio.emit('user_event_name',message)
        # socketio.send(message,namespace='/message')
        time.sleep(1)

# @socketio.on('server_event_name')
# def on_connect(*args,**kwargs):
#     print(args)
#     print(kwargs)
#     pass
#
# @socketio.on('server_event_name')
# def on_disconnect(*args,**kwargs):
#     print(args)
#     print(kwargs)
#     pass

# 动态添加路由
# def custom_func(message):
#     print("custom", message)
# socketio.on_event("custom func", custom_func, namespace="/test")


# 客户端
# import socketio
# from socketio import Namespace
#
# sio = socketio.Client()
#
# # 监听服务端推送消息
# @sio.on(event='user_event_name',namespace='/message')
# def user_event_name(data):
#     print('user_message received with ', data)
#     # sio.emit('my response', {'response': 'my response'})
#
# # 连接服务端 IP+端口
# sio.connect('http://localhost:80',namespaces=['/message'])
# print("000")
#
# # 向服务端发送消息
# sio.emit('server_event_name', data=None,namespace='/message')
# sio.wait()
