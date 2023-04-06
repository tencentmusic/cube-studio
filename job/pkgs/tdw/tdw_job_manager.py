# -*- coding: utf-8 -*-

import os
import time

from ..exceptions.tdw_exceptions import TDWFailedException, TDWNoResException
from .idex_api import IdexApi


class TDWJobManager(object):
    def __init__(self, tdw_sec_file, env="prod", proxy_servers=None):
        """
        封装TDW任务（通过idex openapi）

        Args:
            tdw_sec_file (str): tdw秘钥文件，可从https://tdwsecurity.oa.com/user/keys 下载
            env (str, optional): prod或者dev，分别表示正式和测试环境. Defaults to "prod".
            proxy_servers (dict, optional): 代理服务器地址，{'http': 'xxx', 'https': 'yyy'}. Defaults to None.
        """
        super().__init__()
        self.idex_api = IdexApi(tdw_sec_file, env, proxy_servers)
    
    def run_job(self, database, sql, params=None, export_file=None, write_chunk_size=2**19):
        """
        运行tdw sql任务，export_file为None时，不需要导出结果到文件，否则指定要写入的文件。
        注意只会导出sql中最后一条语句的结果到文件！！！

        Args:
            database (str): 数据库名，任意一个自己有权限的数据库名即可，不一定与要执行的sql相关
            sql (str): 要运行的sql，可以包含多条语句，每条语句后必须有";"
            params (dict, optional): 参数表，对sql内容进行参数替换，sql中的参数应该是${param_name}$格式。 Defaults to None
                                    e.g. {
                                        "param1": "123",
                                        "param2": "'a'"
                                    }
                                    则sql中的${param1}$会被替换为123，而${param2}$会被替换为'a'.
            export_file (str, optional): 如果不为None，则指定要写入的结果文件. Defaults to None.
            write_chunk_size (int, optional): 从网络读取数据块并写入文件时的chunk size, Defaults to 512kb

        Raises:
            TDWNoResException: [description]
            TDWFailedException: [description]
        """
        if not sql or not sql.strip():
            print("sql is empty, will do nothing")
            return
        # 参数替换
        if params:
            for p, v in params.items():
                sql = sql.replace('${'+p+'}$', v)

        max_retries = 5

        def _run(cluster_id, group_id, gaia_id):
            # 创建任务
            st = time.time()
            task_id = self.idex_api.run_sqls(cluster_id, group_id, gaia_id, database, sql)
            if not task_id:
                raise TDWFailedException("create TDW task error, cluster_id='{}', pool_id='{}', group_id='{}', \
                                         gaia_id='{}', sql='{}'".format(cluster_id, pool_id, group_id, gaia_id, sql))
            print("created TDW task '{}', cluster_id='{}', pool_id='{}', group_id='{}', gaia_id='{}', sql='{}'"
                  .format(task_id, cluster_id, pool_id, group_id, gaia_id, sql))

            # 按文档示例，先等10s，再开始查状态（文档中是等5s，这里多等5s）
            retries = 0
            time.sleep(10)
            while True:
                task_status = self.idex_api.get_task_status(task_id)
                if not task_status or not task_status.strip():
                    if retries >= max_retries:
                        break
                    print("failed to get status of TDW task '{}', will retry after 10s ({}/{})"
                          .format(task_id, retries, max_retries))
                    retries += 1
                    time.sleep(10)
                    continue
                retries = 0
                task_status = task_status.strip().lower()
                if task_status in ['running', 'submitted']:
                    print("waiting TDW task '{}' to finish...('{}')".format(task_id, task_status))
                    time.sleep(10)
                else:
                    print("TDW task '{}' status changed to '{}'".format(task_id, task_status))
                    break

            if task_status != 'success':
                raise TDWFailedException("TDW task '{}' failed('{}'), cluster_id='{}', pool_id='{}', group_id='{}', \
                    gaia_id='{}', sql='{}'".format(task_id, task_status, cluster_id, pool_id, group_id, gaia_id, sql))

            print("TDW task '{}' success, cost {}s".format(task_id, time.time()-st))

            if export_file:
                retries = 0
                # 把结果写入文件
                while retries <= max_retries:
                    stmt_ids = self.idex_api.get_task_statement_list(task_id)
                    if not stmt_ids:
                        print("failed to get statement list of TDW task '{}', will retry after 10s ({}/{})"
                              .format(task_id, retries, max_retries))
                        retries += 1
                        time.sleep(10)
                        continue
                    break
                else:
                    raise TDWFailedException("found no statement of TDW task '{}'".format(task_id))
                stmt_id = stmt_ids[-1]
                # 只写最后一个语句的输出
                time.sleep(5)   # 先间隔5秒
                st = time.time()
                print("begin writting result of TDW task '{}' into file '{}'".format(task_id, export_file))
                if not self.idex_api.write_task_statement_result_to_file(task_id, stmt_id, export_file,
                                                                         write_chunk_size):
                    raise TDWFailedException("write result of TDW task '{}' to '{}' error"
                                             .format(task_id, export_file))

                print("wrote result of TDW task '{}' into file '{}' finished, cost {}s"
                      .format(task_id, export_file, time.time()-st))

        # 获取集群列表，并选取集群
        clusters = self.idex_api.get_cluster_list()
        if not clusters:
            raise TDWNoResException("found no TDW cluster")

        for i, cluster_id in enumerate(clusters):
            # 获取资源池列表，并选取资源池
            pools = self.idex_api.get_cluster_res_pool_list(cluster_id)
            if not pools:
                print("found no resource pool of {}th/{} TDW cluster '{}'".format(i+1, len(clusters), cluster_id))
                if i == len(clusters)-1:
                    raise TDWNoResException("found no resource pool of all TDW clusters")
                continue

            for j, pool_id in enumerate(pools):
                pool_info = self.idex_api.get_cluster_res_pool_info(cluster_id, pool_id)
                if not pool_info:
                    print("{}th/{} TDW resource pool '{}' of cluster '{}' not valid"
                          .format(j+1, len(pools), pool_id, cluster_id))
                    if j == len(pools)-1 and i == len(clusters)-1:
                        raise TDWNoResException("all TDW resource pool are not valid")
                    continue
                if 'privacy' in pool_info['group_id'] or 'share' in pool_info['group_id']:
                    print("不选封闭域的应用组，封闭域数据别在这里导出")
                    continue

                group_id = pool_info['group_id']
                gaia_id = pool_info['gaia_id']

                try:
                    _run(cluster_id, group_id, gaia_id)
                except Exception as e:
                    if j == len(pools)-1 and i == len(clusters)-1:
                        raise e

    def run_job_from_file(self, database, sql_file, params=None, export_file=None, write_chunk_size=2**19):
        """
        从sql文件运行tdw sql任务，export_file为None时，不需要导出结果到文件，否则指定要写入的文件。
        注意只会导出sql中最后一条语句的结果到文件！！！

        Args:
            database (str): 数据库名，任意一个自己有权限的数据库名即可，不一定与要执行的sql相关
            sql_file (str): 要运行的sql文件，可以包含多条语句，每条语句后必须有";"
            params (dict, optional): 参数表，对sql内容进行参数替换，sql中的参数应该是${param_name}$格式。 Defaults to None
                                    e.g. {
                                        "param1": "123",
                                        "param2": "'a'"
                                    }
                                    则sql中的${param1}$会被替换为123，而${param2}$会被替换为'a'.
            export_file (str, optional): 如果不为None，则指定要写入的结果文件. Defaults to None.
            write_chunk_size (int, optional): 从网络读取数据块并写入文件时的chunk size, Defaults to 512kb

        Raises:
            TDWNoResException: [description]
            TDWFailedException: [description]
        """
        sql = self.read_sql_file(sql_file)
        if not sql:
            print("read no sql statements from '{}', ignore!".format(sql_file))
            return
        self.run_job(database, sql, params, export_file, write_chunk_size)

    @staticmethod
    def read_sql_file(sql_file):
        """
        读取sql文件，去掉换行符，去掉注释内容

        Args:
            sql_file (str): sql文件

        Returns:
            str: 读取的sql内容
        """
        if not os.path.isfile(sql_file):
            print("'{}' is not valid sql file".format(sql_file))
            return None
        
        flatten_text = ""
        in_comment = 0
        with open(sql_file) as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                if in_comment > 0:
                    # 注释内容
                    continue
                if line.startswith("--"):
                    # 单行注释
                    continue
                if line.startswith("/*"):
                    # 多行注释开始
                    in_comment += 1
                    continue
                if line.endswith("*/"):
                    # 多行注释结束
                    in_comment -= 1
                    continue
                flatten_text += " " + line
        
        return flatten_text.strip()
