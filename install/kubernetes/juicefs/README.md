### 如何以juicefs作为cube-stuido的训练、部署pv的共享目录：
- 1，修改目录(juicefs)中的.env文件，将JUICEFS_HOST_IP改为自己部署cube-studio的节点的ip地址；
- 2，在控制台执行sh start_juicefs.sh