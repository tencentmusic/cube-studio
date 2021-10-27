#!/bin/bash

#set -ex

rm -rf /home/myapp/myapp/static/assets
ln -s /home/myapp/myapp/assets /home/myapp/myapp/static/
rm -rf /home/myapp/myapp/static/appbuilder/mnt
ln -s /data/k8s/kubeflow/global/static /home/myapp/myapp/static/appbuilder/mnt


if [ "$STAGE" = "init" ]; then
  export FLASK_APP=myapp:app
  myapp fab create-admin --username admin --firstname admin --lastname admin --email admin@tencent.com --password admin
  # myapp db init    # 生成migrations文件夹
  # myapp db migrate   # 生成对应版本数据库表的升级文件到versions文件夹下，需要你的数据库是已经upgrade的
  myapp db upgrade   # 数据库表同步更新到mysql
  # 会创建默认的角色和权限。会创建自定义的menu权限，也才能显示自定义menu。
  myapp init

elif [ "$STAGE" = "build" ]; then
  cd /home/myapp/myapp/vision && yarn && yarn build

elif [ "$STAGE" = "dev" ]; then
  export FLASK_APP=myapp:app
#  FLASK_ENV=development  flask run -p 80 --with-threads  --host=0.0.0.0
  python myapp/run.py

elif [ "$STAGE" = "prod" ]; then
  export FLASK_APP=myapp:app
  gunicorn --bind  0.0.0.0:80 --workers 20 --timeout 300 --limit-request-line 0 --limit-request-field_size 0 --log-level=info myapp:app
else
    myapp --help
fi


