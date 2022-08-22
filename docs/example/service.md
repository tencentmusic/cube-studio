
# 内部服务

##  普通服务

### 开发注册

1、开发你的服务化镜像，push到docker仓库内

2、注册你的服务

![image](https://user-images.githubusercontent.com/20157705/169932303-0ec981cc-09ca-423c-96f9-da164ed309da.png)

## mysql web服务

镜像：ccr.ccs.tencentyun.com/cube-studio/phpmyadmin

环境变量：
```
PMA_HOST=xx.xx.xx.xx
PMA_PORT=xx
PMA_USER=xx
PMA_PASSWORD=xx
```
端口：80

## mongo web服务
镜像：mongo-express:0.54.0

环境变量：
```
ME_CONFIG_MONGODB_SERVER=xx.xx.xx.xx
ME_CONFIG_MONGODB_PORT=xx
ME_CONFIG_MONGODB_ENABLE_ADMIN=true
ME_CONFIG_MONGODB_ADMINUSERNAME=xx
ME_CONFIG_MONGODB_ADMINPASSWORD=xx
ME_CONFIG_MONGODB_AUTH_DATABASE=xx
VCAP_APP_HOST=0.0.0.0
VCAP_APP_PORT=8081
ME_CONFIG_OPTIONS_EDITORTHEME=ambiance
```
端口：8081

## redis web
镜像：ccr.ccs.tencentyun.com/cube-studio/patrikx3:latest

环境变量
```
REDIS_NAME=xx
REDIS_HOST=xx
REDIS_PORT=xx
REDIS_PASSWORD=xx
```
端口：7843

## 图数据库neo4j

镜像：ccr.ccs.tencentyun.com/cube-studio/neo4j:4.4

环境变量
```
NEO4J_AUTH=neo4j/admin
```
端口：7474,7687

## jaeger链路追踪

镜像：jaegertracing/all-in-one:1.29

端口：5775,16686


## 服务暴露：

 - 1、域名暴露需要平台配置泛域名，SERVICE_DOMAIN
 - 2、ip暴露需要平台配置SERVICE_EXTERNAL_IP=[xx.xx.xx.xx]，或者项目中配置SERVICE_EXTERNAL_IP=xx.xx.xx.xx  ip需为集群中节点的ip或者clb的ip

