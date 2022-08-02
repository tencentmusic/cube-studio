# /bin/bash

# 设置多个环境变量到 environment variable
# echo -e "set \$variable1 $PATH;
# set \$variable2 $PATH;
# set \$variable3 $PATH;" > /opt/aaa/env.variable;

# 设置单个环境变量到 environment variable
#echo set \$APP_ROOT $APP_ROOT\; > /etc/nginx/conf.d/env.variable
sed -i 's/APP_ROOT/${APP_ROOT}/' /etc/nginx/conf.d/default.conf
# 启动 nginx 应用
nginx -g "daemon off;"
# 防止容器启动后进程退出，导致容器退出；实现容器后台运行
sh
