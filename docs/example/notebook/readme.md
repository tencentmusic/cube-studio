# 在线notebook开发

### notebook支持类型
1. Jupyter （cpu/gpu）
2. vscode（cpu/gpu）

### 支持功能
1. 代码开发/调试，上传/下载，命令行，git工蜂/github，内网/外网，tensorboard，自主安装插件

### 添加notebook
路径：在线开发->notebook->添加

![](../pic/tapd_20424693_1630648630_29.png)

备注：
1. Running状态下，方可进入
2. 无法进去时，reset重新激活
3. notebook会自动挂载一下目录
 - a）个人工作目录到容器 /mnt/$username
 - b）个人归档目录到容器/archives/$username

### jupyter示例：

![](../pic/tapd_20424693_1611142088_15.png)

### vscode示例：

![](../pic/tapd_20424693_1615455976_85.png)

### 切换归档目录示例：

![](../pic/tapd_20424693_1619156218_75.png)

### tensorboard示例：

进入到对应的日志目录，再打开tensorboard按钮

![](../pic/tapd_20424693_1630381219_76.png)
