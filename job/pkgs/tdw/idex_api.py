# -*- coding: utf-8 -*-

import json
import traceback

import requests
import time
import socket

from .tdw_tauth_authentication import TdwTauthAuthentication

API_BASE_URL = "http://api.idex.oa.com"
API_TARGET_NAME = "idex-openapi"


class IdexApi(object):
    
    def __init__(self, tdw_sec_file, env="prod", proxy_servers=None):
        """
        对idex openapi的封装
        Args:
            tdw_sec_file (str): tdw秘钥文件，可从https://tdwsecurity.oa.com/user/keys 下载
            env (str, optional): prod或者dev，分别表示正式和测试环境. Defaults to "prod".
            proxy_servers (dict, optional): 代理服务器地址，{'http': 'xxx', 'https': 'yyy'}. Defaults to None.
        """
        super().__init__()
        self.tta = TdwTauthAuthentication(tdw_sec_file, API_TARGET_NAME, env=env, proxyServers=proxy_servers)
        self.req_headers = {"Version": "2"}
        self.proxy_servers = proxy_servers
        
    def get_cluster_list(self):
        """
        获取集群列表， 见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e9%9b%86%e7%be%a4%e5%88%97%e8%a1%a8
        
        Returns:
            list: 集群id列表
        """
        api_url = API_BASE_URL + "/clusters"
        try:
            response = self._request('get', api_url)
            if response.status_code // 100 > 2:
                print("failed to get cluster list, api_url='{}', host='{}', response=[{}, {}, {}]"
                      .format(api_url, self.__get_api_host(), response.status_code, response.reason,
                              response.text))
                return []

            url_list = self.__parse_response_content(response)
            id_list = list(map(lambda x: x.split('/')[-1], url_list))
            return id_list
        except Exception as e:
            print("get cluster list error, api_url='{}', host='{}', resquest headers={}: {}\n{}"
                  .format(api_url, self.__get_api_host(), socket.gethostbyname(''), e, traceback.format_exc()))
            return None
    
    def get_cluster_info(self, cluster_id):
        """
        获取单个集群信息，见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e5%8d%95%e4%b8%aa%e9%9b%86%e7%be%a4

        Args:
            cluster_id (str): 集群id，从get_cluster_list结果中取得

        Returns:
            dict: 集群信息，
            e.g. {
                "name": "同乐", 
                "pools_url": "http://api.idex.oa.com/clusters/tl/pools"
            }
        """
        api_url = API_BASE_URL + "/clusters/" + cluster_id
        try: 
            response = self._request('get', api_url)
            if response.status_code // 100 > 2:
                print("failed to get info of cluster '{}', api_url='{}', host='{}', response=[{}, {}, {}]"
                      .format(cluster_id, api_url, self.__get_api_host(), response.status_code, response.reason,
                              response.text))
                return {}
            return self.__parse_response_content(response)
        except Exception as e:
            print("get info of cluster '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(cluster_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
    
    def get_cluster_res_pool_list(self, cluster_id):
        """
        获取集群资源池列表，见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e8%b5%84%e6%ba%90%e6%b1%a0%e5%88%97%e8%a1%a8

        Args:
            cluster_id (str): 集群id，从get_cluster_list结果中取得

        Returns:
            dict: 资源池id列表
        """
        api_url = API_BASE_URL + "/clusters/" + cluster_id + "/pools"
        try:
            response = self._request('get', api_url)
            if response.status_code // 100 > 2:
                print("failed to get resource pool list of cluster '{}', api_url='{}', host='{}',"
                      " response=[{}, {}, {}]".format(cluster_id, api_url, self.__get_api_host(),
                                                      response.status_code, response.reason,
                                                      response.text))
                return []
            url_list = self.__parse_response_content(response)
            id_list = list(map(lambda x: x.split('/')[-1], url_list))
            return id_list
        except Exception as e:
            print("get resource pool list of cluster '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(cluster_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
    
    def get_cluster_res_pool_info(self, cluster_id, pool_id):
        """
        获取资源池信息，见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e5%8d%95%e4%b8%aa%e8%b5%84%e6%ba%90%e6%b1%a0

        Args:
            cluster_id (str): 集群id，从get_cluster_list结果中取得
            pool_id (str): 资源池id，从get_cluster_res_pool_list结果中取得
        Returns:
            dict: 资源池信息
            e.g. {
                "gaia_id": "573", 
                "group_id": "g_teg_tdw_idex", 
                "name": "g_teg_tdw_idex 同乐盖亚"
            }
        """
        api_url = API_BASE_URL + "/clusters/" + cluster_id + "/pools/" + pool_id
        try:
            response = self._request('get', api_url)
            if response.status_code // 100 > 2:
                print("failed to get info of resource pool '{}' of cluster '{}', api_url='{}', host='{}',"
                      " response=[{}, {}, {}]".format(pool_id, cluster_id, api_url, self.__get_api_host(),
                                                      response.status_code, response.reason, response.text))
                return {}
            return self.__parse_response_content(response)
        except Exception as e:
            print("get info of resource pool '{}' of cluster '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(pool_id, cluster_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
    
    def run_sqls(self, cluster_id, group_id, gaia_id, database, statements):
        """
        运行单个任务，见https://idex.oa.com/help/#/Open_API?id=%e8%bf%90%e8%a1%8c%e5%8d%95%e4%b8%aa%e4%bb%bb%e5%8a%a1

        Args:
            cluster_id (str): 集群id
            group_id (str): group_id，从get_cluster_res_pool_info结果中取得
            gaia_id (str): gaia_id，从get_cluster_res_pool_info结果中取得
            database (str): 数据库名，任意一个自己有权限的数据库名即可，不一定与要执行的sql相关
            statements (str): 要运行的sql，可以包含多条语句，每条语句后必须有";"

        Returns:
            str: 任务id
        """
        api_url = API_BASE_URL + "/tasks"
        data = {
            "cluster_id": cluster_id,
            "group_id": group_id,
            "gaia_id": gaia_id,
            "database": database,
            "statements": statements,
            "mode": "full"
        }
        try:
            response = self._request('post', api_url, data=data)
            if response.status_code // 100 > 2:
                print("failed to run sqls, api_url='{}', host='{}', data={}, response=[{}, {}, {}]"
                      .format(api_url, self.__get_api_host(), data, response.status_code, response.reason,
                              response.text))
                return ""
            body_json = self.__parse_response_content(response)
            task_url = body_json.get('task_url')
            return task_url.split('/')[-1]
        except Exception as e:
            print("run sqls error, data='{}', api_url='{}', host='{}': {}\n{}"
                  .format(data, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
    
    def get_task_status(self, task_id):
        """
        获取单个任务信息，见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e5%8d%95%e4%b8%aa%e4%bb%bb%e5%8a%a1

        Args:
            task_id (str): 任务id，从run_sqls获得

        Returns:
            str: 任务状态，包括 running、success 和 abortion, failure 等
        """
        api_url = API_BASE_URL + "/tasks/" + task_id
        try:
            response = self._request('get', api_url)
            if response.status_code // 100 == 4:
                print("task '{}' not found, api_url='{}', host='{}', response=[{}, {}, {}]"
                      .format(task_id, api_url, self.__get_api_host(), response.status_code,
                              response.reason, response.text))
                return "missed"
            elif response.status_code // 100 > 2:
                print("failed to get info of task '{}', api_url='{}', host='{}', response=[{}, {}, {}]"
                      .format(task_id, api_url, self.__get_api_host(), response.status_code, response.reason,
                              response.text))
                return ""
            info = self.__parse_response_content(response)
            return info['state']
        except Exception as e:
            print("get info of task '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(task_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
    
    def kill_task(self, task_id):
        """
        终止任务，见https://idex.oa.com/help/#/Open_API?id=%e7%bb%88%e6%ad%a2%e5%8d%95%e4%b8%aa%e4%bb%bb%e5%8a%a1

        Args:
            task_id (str): 任务id，从run_sqls获得
        """
        api_url = API_BASE_URL + "/tasks/" + task_id
        try:
            response = self._request('delete', api_url)
            if response.status_code // 100 > 2:
                print("failed to kill task '{}', api_url='{}', host='{}', response=[{}, {}, {}]"
                      .format(task_id, api_url, self.__get_api_host(), response.status_code, response.reason,
                              response.text))
                return False
            return True
        except Exception as e:
            print("kill task '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(task_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return False
        
    def get_task_statement_list(self, task_id):
        """
        获取任务的语句id列表，见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e8%af%ad%e5%8f%a5%e5%88%97%e8%a1%a8

        Args:
            task_id (str): 任务id，从run_sqls获得

        Returns:
            list: 语句id列表
        """
        api_url = API_BASE_URL + "/tasks/" + task_id + "/statements"
        try:
            response = self._request("get", api_url)
            if response.status_code // 100 > 2:
                print("failed to get statements of task '{}', api_url='{}', host='{}', response=[{}, {}, {}]"
                      .format(task_id, api_url, self.__get_api_host(), response.status_code, response.reason,
                              response.text))
                return None
            url_list = self.__parse_response_content(response)
            id_list = list(map(lambda x: x.split('/')[-1], url_list))
            return id_list
        except Exception as e:
            print("get statements of task '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(task_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
    
    def get_task_statement_info(self, task_id, statement_id):
        """
        获取单个语句信息，见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e5%8d%95%e4%b8%aa%e8%af%ad%e5%8f%a5

        Args:
            task_id (str): 任务id，从run_sqls获得
            statement_id (str): 语句id，从get_task_statement_list结果中取得

        Returns:
            dict: 语句信息
            e.g. {
                "jobs_url": "http://application.tdw.oa.com:8080/proxy/application_1587540414210_89628", 
                "result_url": "http://api.idex.oa.com/tasks/6d093449-b3ce-424b-800e-09a37f8cdb4f/statements/plc_1595246537863_9585/result", 
                "state": "success"
            }
            其中jobs_url是作业链接，state是状态，包括 running、success 和 abortion, failure 等
        """
        api_url = API_BASE_URL + "/tasks/" + task_id + "/statements/" + statement_id
        try:
            response = self._request('get', api_url)
            if response.status_code // 100 == 4:
                print("statement '{}' of task '{}' not found, api_url='{}', host='{}', response=[{}, {}, {}]"
                      .format(statement_id, task_id, api_url, self.__get_api_host(), response.status_code,
                              response.reason, response.text))
                return {"state": "missed"}
            elif response.status_code // 100 > 2:
                print("failed to get info of statement '{}' of task '{}', api_url='{}', host='{}',"
                      " response=[{}, {}, {}]".format(statement_id, task_id, api_url, self.__get_api_host(),
                                                      response.status_code, response.reason, response.text))
                return {}
            return self.__parse_response_content(response)
        except Exception as e:
            print("get info of statement '{}' of task '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(statement_id, task_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
    
    def get_task_statement_result(self, task_id, statement_id):
        """
        获取单个语句的执行结果，见https://idex.oa.com/help/#/Open_API?id=%e8%8e%b7%e5%8f%96%e7%bb%93%e6%9e%9c

        Args:
            task_id (str): 任务id，从run_sqls获得
            statement_id (str): 语句id，从get_task_statement_list结果中取得

        Returns:
            str: 语句的执行结果，不同语句的执行结果不一样
        """
        api_url = API_BASE_URL + "/tasks/" + task_id + "/statements/" + statement_id + "/result"
        try:
            response = self._request('get', api_url)
            if response.status_code // 100 > 2:
                print("failed to get result of statement '{}' of task '{}', api_url='{}', host='{}',"
                      " response=[{}, {}, {}]".format(statement_id, task_id, api_url, self.__get_api_host(),
                                                      response.status_code, response.reason, response.text))
                return ""
            return response.text
        except Exception as e:
            print("get result of statement '{}' of task '{}' error, api_url='{}', host='{}': {}\n{}"
                  .format(statement_id, task_id, api_url, self.__get_api_host(), e, traceback.format_exc()))
            return None
        
    def write_task_statement_result_to_file(self, task_id, statement_id, file_name, chunk_size=2**19):
        """
        把语句执行的结果写入文件

        Args:
            task_id (str): 任务id，从run_sqls获得
            statement_id (str): 语句id，从get_task_statement_list结果中取得
            file_name (str): 目标文件路径
            chunk_size (int, optional): 每次写入大小. Defaults to 2**19.

        Returns:
            bool: True表示写成功，False表示写失败
        """
        api_url = API_BASE_URL + "/tasks/" + task_id + "/statements/" + statement_id + "/result"
        try:
            sess = requests.session()
            with sess:
                with sess.request('get', api_url, headers=self._make_headers(), proxies=self.proxy_servers,
                                  stream=True) as response:
                    if response.status_code // 100 > 2:
                        print("failed to get result of statement '{}' of task '{}', api_url='{}', host='{}',"
                              " response=[{}, {}, {}]".format(statement_id, task_id, api_url, self.__get_api_host(),
                                                              response.status_code, response.reason, response.text))
                        return False
                    with open(file_name, 'wb', buffering=2**25) as f:
                        max_retry = 10
                        tries = 1
                        while tries <= max_retry:
                            try:
                                print("force chunk_size=None")
                                for chunk in response.iter_content(chunk_size=None):
                                    if chunk:
                                        f.write(chunk)
                                    else:
                                        print("got empty chunk of statement '{}' of task '{}'"
                                              .format(statement_id, task_id))
                            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) \
                                    as e1:
                                retry_wait_time = min(16, 2**(tries-1))
                                retry_msg = "will retry after {}s".format(retry_wait_time) \
                                    if tries < max_retry else "abort"
                                print("{}/{} try of receiving content of statement '{}' of task '{}' encounter error,"
                                      " {}: {}\n{}".format(tries, max_retry, statement_id, task_id, retry_msg, e1,
                                                           traceback.format_exc()))
                                if tries < max_retry:
                                    time.sleep(retry_wait_time)
                                else:
                                    raise e1
                                tries += 1
                                continue
                            break
                return True
        except Exception as e:
            print("write result of statement '{}' of task '{}' into file '{}' error, chunk_size={}, api_url='{}',"
                  " host='{}': {}\n{}".format(statement_id, task_id, file_name, chunk_size, api_url,
                                              self.__get_api_host(), e, traceback.format_exc()))
            return False
        
    def _request(self, method, url, params=None, data=None):
        return requests.request(method, url, params=params, json=data, headers=self._make_headers(), 
                                proxies=self.proxy_servers)
        
    def _make_headers(self):
        auth = self.tta.getAuthentication()
        auth_str = json.dumps(auth)
        self.req_headers["Authentication"] = auth_str
        return self.req_headers

    @classmethod
    def __parse_response_content(cls, resp: requests.Response):
        if not resp:
            raise RuntimeError("response can not be None")
        try:
            json_ctnt = json.loads(resp.text)
            return json_ctnt
        except Exception as e:
            raise RuntimeError("parse response content error: host='{}', code={}, reason='{}', text='{}', headers={},"
                               " request headers={}".format(cls.__get_api_host(), resp.status_code, resp.reason,
                                                            resp.text, resp.headers,
                                                            None if resp.request is None else resp.request.headers))

    @classmethod
    def __get_api_host(cls):
        import socket
        import urllib3
        _, domain, _ = urllib3.get_host(API_BASE_URL)
        return domain, socket.gethostbyname(domain)
