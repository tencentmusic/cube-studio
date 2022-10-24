
import argparse
import datetime
import gettext
import io
import json
import logging
import os
import re
import sys
import time
import click
import shutil

def create_app(app):

    os.mkdir(app)
    app_name = app[0:1].upper() + app[1:]

    version = datetime.datetime.now().strftime("%Y%m%d")
    app_py = '''
import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type

import pysnooper
import os

class {app_name}_Model(Model):
    # 模型基础信息定义
    name='{app}'
    label='{app}应用中文名'
    description="{app}应用，详细描述，都会显示应用描述上，或者添加一些说明"
    field="机器视觉"
    scenes="图像识别"
    status='online'
    version='v{version}'
    doc='https://帮助文档的链接地址'
    pic='https://应用描述的缩略图/可以直接使用应用内的图片文件地址'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='推理函数的输入参数', describe='输入函数的详细描述')
    ]

    # 加载模型
    def load_model(self):
        self.model = load("/xxx/xx/a.pth")

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        result_img='result/target.jpg'
        result_text='cat,dog'
        back=[{{
            "image":result_img,
            "text":result_text
        }}]
        return back

model={app_name}_Model()
model.load_model()
result = model.inference(img_file_path='test.jpg')  # 测试
print(result)

# 启动服务
server = Server(model=model)
server.web_examples.append({{
    "img_file_path":"test.jpg"
}})
server.server(port=8080)
                    '''.format(app=app, app_name=app_name, version=version)
    file = open(app + "/app.py", mode='w')
    file.write(app_py)
    file.close()

    dockerfile = f'''
# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:{app}  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base
# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh
# 安装文件
WORKDIR /app
COPY * /app/
ENTRYPOINT ["python", "app.py"]
            '''
    file = open(app + "/Dockerfile", mode='w')
    file.write(dockerfile)
    file.close()

    init = f'''
# apt install xx
# pip install xx
            '''
    file = open(app + "/init.sh", mode='w')
    file.write(init)
    file.close()

    shutil.copyfile(os.path.dirname(os.path.abspath(__file__)) + '/test.jpg', f'{app}/test.jpg')
    print(f'create app {app} success, cd {app}, 执行 cube debug,进入开发')
    pass


@click.command()
@click.argument('ops', required=True)
@click.argument('app', required=False)
@click.option('-p', '--port', default=8080, help='暴露端口')
def main(ops,port,app):
    print(ops,port,app)
    if ops=='debug':
        app = os.getcwd()[os.getcwd().rindex("/")+1:]
        cube_dir = os.path.dirname(os.path.dirname(os.getcwd()))+"/src"
        command =f'docker run --name {app} --privileged -it -v {cube_dir}:/src -v $PWD:/app -p 8080:8080  ccr.ccs.tencentyun.com/cube-studio/aihub:base bash'

        print('1、使用如下命令启动开发环境：\n',command)
        print('2、完善init.sh初始化环境脚本，和app.py应用脚本')
        print('3、最后使用下列命令打包镜像\ndocker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:%s .'%app)
    if ops=='deploy':
        app = os.getcwd()[os.getcwd().rindex("/")+1:]
        command =f'docker run --name {app} --privileged -it -v $cube_dir/src:/src -v $PWD:/app -p 8080:8080  ccr.ccs.tencentyun.com/cube-studio/aihub:{app}'
        print(command)
    if ops=='create' and app:
        app = app.lower()
        if os.path.exists(app):
            print("app: %s exist" % app)
            return
        create_app(app)

if __name__ == "__main__":
    main()