#!/usr/bin/env python
# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# pip install psycopg2-binary pymysql

import os
import shutil

import pysnooper

@pysnooper.snoop()
def init_db(SQLALCHEMY_DATABASE_URI):
    if SQLALCHEMY_DATABASE_URI:
        import sqlalchemy.engine.url as url
        uri = url.make_url(SQLALCHEMY_DATABASE_URI)
        """Inits the Myapp application"""
        sql = 'CREATE DATABASE IF NOT EXISTS example'
        if 'mysql' in SQLALCHEMY_DATABASE_URI:
            import pymysql
            # 创建连接
            conn = pymysql.connect(host=uri.host, port=uri.port, user=uri.username, password=uri.password, charset='utf8')
            sql = "CREATE DATABASE IF NOT EXISTS example DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;"
        elif 'postgre' in SQLALCHEMY_DATABASE_URI:
            import psycopg2
            # 创建连接
            conn = psycopg2.connect(host=uri.host, port=uri.port, user=uri.username, password=uri.password)
            sql = "CREATE DATABASE example WITH ENCODING 'UTF8'"
        # 创建游标
        cursor = conn.cursor()
        conn.autocommit = True
        # 创建数据库的sql(如果数据库存在就不创建，防止异常)

        print(sql)
        # 执行创建数据库的sql
        cursor.execute(sql)
        conn.commit()


# 将文件导入到mysql数据库
@pysnooper.snoop()
def csv_table(csv_path,SQLALCHEMY_DATABASE_URI):
    # -*- coding:UTF-8 -*-

    import pandas as pd
    from sqlalchemy import create_engine
    import sqlalchemy.engine.url as url
    uri = url.make_url(SQLALCHEMY_DATABASE_URI)
    # 数据库信息
    setting = {
        'host': uri.host,
        'port': uri.port,
        'user': uri.username,
        'passwd': uri.password,
        # 数据库名称
        'db': 'example',
        'charset': 'utf8'
    }
    # 表名
    # 如果不存在表，则自动创建
    table_name = os.path.basename(csv_path).replace('.csv','')
    # 文件路径
    path = csv_path

    data = pd.read_csv(path, encoding='utf-8')
    print(data)
    if 'mysql' in SQLALCHEMY_DATABASE_URI:
        engine = create_engine("mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}".format(**setting), max_overflow=5)
    elif 'postgre' in SQLALCHEMY_DATABASE_URI:
        engine = create_engine("postgresql+psycopg2://{user}:{passwd}@{host}:{port}/{db}".format(**setting), max_overflow=5)
    data.to_sql(table_name, engine, index=False, if_exists='replace', )
    print('导入成功...')

def init():
    mysql_sql_uri='mysql+pymysql://root:admin@mysql-service.infra:3306/example?charset=utf8'
    postgres_sql_uri = 'postgresql+psycopg2://postgres:postgres@postgresql.kubeflow:5432/example'
    current_work_dir = os.path.dirname(__file__)
    try:
        init_db(SQLALCHEMY_DATABASE_URI=mysql_sql_uri)
        csv_table(csv_path=os.path.join(current_work_dir, 'train.csv'), SQLALCHEMY_DATABASE_URI=mysql_sql_uri)
    except Exception as e:
        print(e)
    try:
        init_db(SQLALCHEMY_DATABASE_URI=postgres_sql_uri)
        csv_table(csv_path=os.path.join(current_work_dir,'train.csv'),SQLALCHEMY_DATABASE_URI=postgres_sql_uri)
    except Exception as e:
        print(e)

init()


