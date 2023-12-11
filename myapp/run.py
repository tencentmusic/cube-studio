from myapp import app
app.run(host="0.0.0.0", port=80, debug=True)

# from myapp import socketio
# socketio.run(app,host="0.0.0.0", port=80, debug=True)

# from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
# server = pywsgi.WSGIServer(('0.0.0.0', 80), app, handler_class=WebSocketHandler)
# server.serve_forever()

# #配置websocket
# from geventwebsocket.handler import WebSocketHandler
# from gevent.pywsgi import WSGIServer
# app.debug=True
# # 如果是http请求走app使用原有的wsgi处理，如果是websocket请求走WebSocketHandler处理
# http_server = WSGIServer(('0.0.0.0',80), app, handler_class=WebSocketHandler)
# http_server.serve_forever()







