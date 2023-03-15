# 前缀知识

1、自带环境变量

``
PYTHONPATH=/src/:/github/:$PYTHONPATH
``
其中

/src用于挂载或存放 cubestudio 的python sdk包
/github用于下载存放github上的python sdk包

2、启动命令

cubestudio自带了前端，整体的启动命令
```bash
/src/docker/entrypoint.sh python app.py
```
其中entrypoint.sh 中先安装了必要的库，然后启动前端，然后再启动python app.py

# app.py开发

开发调试模型训练和推理是

1、启动内容

在app.py内，统一使用model.run()启动，会根据用户的命令行启动方式识别具体做什么工作

```bash
python app.py train --arg1 xx --arg2 xx        启动训练，会调用Model的train方法，该方法必须将模型文件保存到save_model_dir指定的目录下
python app.py web --save_model_dir xx          启动web服务，对接load_mode方法和inference方法
```

2、Model class 内的配置

参考app1/app.py，注意代码中的注释

