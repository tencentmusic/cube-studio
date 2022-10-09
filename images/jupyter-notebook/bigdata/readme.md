## 一、组件版本(自行更改)
| **组件名称** | **组件版本** | **备注** |
| --- | --- | --- |
| Hadoop | 3.2.2 | hadoop-3.2.2 |
| Spark | 3.1.3 | spark-3.1.3-bin-hadoop3.2 |
| Hive | 3.1.2 | apache-hive-3.1.2-bin |
| Flink | 1.15.1 | pyFlink |
| Python | 3.8.12 | notebook:jupyter-ubuntu-cpu-base 自带版本 |

## 二、基于Dockerfile实现

#### 2.1 修改Hadoop/hive配置文件
修改conf/hadoop/core-site.xml、conf/hadoop/hdfs-site.xml、conf/hadoop/yarn-site.xml、conf/hive/hive-site.xml对应配置内容
注意检查yarn-site.xml一定要有yarn.resourcemanager.address 配置项，否则默认值是0.0.0.0:8032会导致JupyterLab中的作业无法提交到yarn上运行。示例配置如下:
```xml
<property>
    <name>yarn.resourcemanager.address</name>
    <value>xxx.xxx.xxx.xxx:8032</value>
</property>
```
另外notebook会自带环境变量
```bash
USERNAME=用户名
SERVICE_EXTERNAL_IP=notebook代理ip
PORT1=暴露端口1
PORT2=暴露端口2
```
可以使用这些环境变量来实现对driver的访问和用户的认证

### 2.2 添加测试示例
在example下面添加pyspark  pyflink  clickhouse  impala presto mysql postgresql等数据查询分析示例，或大内存数据分析工具Arrow 以及jupyter中可视化分析方法，供其他数据同学参考使用

平台会自动将example软链到用户个人目录下

### 2.3 通过Dockerfile构建镜像
```bash
docker build -t  $hubhost/notebook:jupyter-ubuntu-cpu-bigdata -f Dockerfile .
docker push $hubhost/notebook:jupyter-ubuntu-cpu-bigdata
```


### 2.4 上线自己的notebook镜像到cube-studio

config.py中 NOTEBOOK_IMAGES 变量为notebook可选镜像，更新此变量即可。

