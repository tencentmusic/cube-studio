# 开发框架
平台控制端为fab框架，可以参考https://github.com/tencentmusic/fab

# 登录方式/首页内容/消息推送
myapp/project.py中包含web首页的配置方式、登录的方式、推送消息的方式

# 定时任务的开发
启动配置：config.py中CeleryConfig
代码开发：myapp/tasks/schedules.py 

# 监听crd变化
代码开发：myapp/tools/watch_xx.py

# 数据库的更新迭代
myapp/migrations/versions

# 数据库结构和视图(增删改查界面)
myapp/models
myapp/views

# pipline编排界面前端
myapp/vision

# 权限管理的基础逻辑
myapp/security.py