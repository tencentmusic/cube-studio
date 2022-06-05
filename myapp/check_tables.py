#!/usr/bin/env python
import os
import pysnooper
# @pysnooper.snoop()
def check_tables():
    SQLALCHEMY_DATABASE_URI = os.getenv('MYSQL_SERVICE','')
    if SQLALCHEMY_DATABASE_URI:
        import sqlalchemy.engine.url as url
        uri = url.make_url(SQLALCHEMY_DATABASE_URI)
        """Inits the Myapp application"""
        import pymysql
        # 创建连接
        conn = pymysql.connect(host=uri.host,port=uri.port, user=uri.username, password=uri.password, charset='utf8')
        # 创建游标
        cursor = conn.cursor()

        # 创建数据库的sql(如果数据库存在就不创建，防止异常)
        sql = "SELECT table_name FROM information_schema.tables  WHERE table_schema='kubeflow'"
        cursor.execute(sql)
        results = list(cursor.fetchall())
        results = [item[0] for item in results]
        print(results)
        for table_name in ['ab_permission', 'ab_permission_view', 'ab_permission_view_role', 'ab_register_user', 'ab_role', 'ab_user', 'ab_user_role', 'ab_view_menu', 'alembic_version', 'docker', 'images', 'inferenceservice', 'job_template', 'kfservice', 'logs', 'nni', 'notebook', 'pipeline', 'project', 'project_user', 'repository', 'run', 'service', 'service_pipeline', 'task', 'user_attribute', 'workflow']:
            if table_name not in results:
                print('kubeflow db下，table %s不完整，请\n1、mysql: drop dabatase kubeflow\n2、重启当前pod'%table_name)
                exit(1)

check_tables()


