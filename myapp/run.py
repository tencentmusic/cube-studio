from myapp import app
# 如果使用 flask run 命令启动，将忽视 这里配置8080，而采用默认的5000端口
app.run(host="0.0.0.0", port=80, debug=True)








