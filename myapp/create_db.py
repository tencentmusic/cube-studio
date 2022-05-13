#!/usr/bin/env python
import os
import pysnooper
# @pysnooper.snoop()
def init_db():
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
        sql = "CREATE DATABASE IF NOT EXISTS kubeflow DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;"
        # 执行创建数据库的sql
        cursor.execute(sql)
        conn.commit()

init_db()

