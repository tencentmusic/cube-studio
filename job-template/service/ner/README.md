# ner service 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/ner:service-20220812

# 参数解析

`--service_type`：服务类型，一般 web 服务镜像填 `serving`。

`--images`：服务镜像，上文第二步打的镜像。

`--ports`：web 镜像里面  rest 服务的端口号，这里填入将其映射出来

# 使用服务

* 点击 IP 访问服务

> 访问地址后面加上`docs`  类似：`http://xx.xx.xx.xx:xx/docs`，可利用 FastAPI 的接口访问服务

* 点击 Try it out ，输入待检测文本
