# coding=utf-8
# @Time     : 2021/7/13 19:33
# @Auther   : lionpeng@tencent.com

import asyncio
import datetime
import re
import time
import traceback
import threading

from polaris.api.consumer import create_consumer_by_config
from polaris.pkg.config.api import Configuration
from polaris.pkg.model.service import GetOneInstanceRequest

from .taf.ef.ai.EFIndexSvr import *
from ..exceptions.ef_exception import *

EF_SERVER_L5_ID = "1881473:196613"


class EFClient(object):
    SERVANT_NAME = "kubeflow.rpcclients.taf.ef.EFIndexSvrProxy"

    INDEX_STATUS_UPDATING = "1"
    INDEX_STATUS_FAILED = "2"
    INDEX_STATUS_SUCCESS = "3"

    def __init__(self, env):
        self.env = env
        config = Configuration()
        config.set_default()
        config.verify()
        self._polaris_api_consumer = create_consumer_by_config(config)

    def _get_rpc_client(self):
        req = GetOneInstanceRequest(namespace=self.env, service=EF_SERVER_L5_ID, use_discover_cache=True)
        if isinstance(threading.current_thread(), threading._MainThread):
            loop = asyncio.get_event_loop()
        else:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            loop = asyncio.get_event_loop()

        ins = loop.run_until_complete(self._polaris_api_consumer.async_get_one_instance(req))
        proxy = EFIndexSvrProxy()
        conn_info = "{}@tcp -h {} -p {}".format(self.SERVANT_NAME, ins.get_host(), ins.get_port())
        print("ef rpc client conn_info='{}'".format(conn_info))
        proxy.locator(conn_info)
        return proxy

    def start_deploy_embedding(self, project_name, model_name, version, is_fallback, vec_file_path,
                               index_id, thr_exp=True):
        client = self._get_rpc_client()
        req = STUpdateEfIndexReq()
        req.project = project_name
        req.model_name = model_name
        req.version = version
        req.index_id = index_id
        req.type = "index"
        m = re.match(r'.*[/\\]+(\d{8})(([/\\].*)|$)', vec_file_path)
        if m:
            req.date = m.group(1)
            print("extracted date '{}' from vec_file_path '{}'".format(req.date, vec_file_path))
        else:
            print("found no date info from vec_file_path '{}'".format(vec_file_path))

        try:
            ret, resp = client.updateIndex(req)
            if ret != 0:
                print("start embedding deployment failed, req={}, ret={}, resp={}"
                      .format(req.toJSON(), ret, resp.toJSON()))
                raise StartDeployEmbeddingError("start embedding deployment error: [{}, {}]".format(ret, resp.toJSON()))
            print("started embedding deployment, req={}, resp={}".format(req.toJSON(), resp.toJSON()))
            return 0
        except Exception as e:
            print("started embedding deployment error: {}\nreq={}\n{}".format(e, req.toJSON(), traceback.format_exc()))
            if thr_exp:
                raise e
            return -1

    def query_deploy_status(self, project_name, model_name, version, thr_exp=True):
        client = self._get_rpc_client()
        req = STGetEmbStatsReq()
        req.project = project_name
        req.model_name = model_name
        req.version = version
        try:
            ret, resp = client.checkEmbStats(req)
            if ret != 0:
                print("query embedding deployment failed, req={}, ret={}, resp={}"
                      .format(req.toJSON(), ret, resp.toJSON()))
                raise QueryEmbeddingDeployError("query embedding deployment error: [{}, {}]".format(ret, resp.toJSON()))

            return resp
        except Exception as e:
            print("query embedding deployment error: {}\nreq={}\n{}".format(e, req.toJSON(), traceback.format_exc()))
            if thr_exp:
                raise e
            return None

    def deploy_embedding_and_wait(self, project_name, model_name, version, is_fallback, vec_file_path,
                                  index_id, wait_time=None):
        max_retry = 10
        for i in range(1, max_retry+1):
            try:
                self.start_deploy_embedding(project_name, model_name, version, is_fallback, vec_file_path,
                                            index_id, True)
                break
            except Exception as e:
                if i == max_retry:
                    raise e
                sleep_time = min(2 ** (i - 1), 60)
                print("{}/{} try of starting embedding deployment failed: {}, will retry after {}s"
                      .format(i, max_retry, e, sleep_time))
                time.sleep(sleep_time)

        if wait_time:
            from ..utils import parse_timedelta
            wait_time = parse_timedelta(wait_time)

        elapsed = 0
        error_times = 0
        interval = 5
        st = time.perf_counter()
        while True:
            try:
                query_ret = self.query_deploy_status(project_name, model_name, version, True)
                status = query_ret.data.get('status')
                if status == self.INDEX_STATUS_SUCCESS:
                    print("embedding deployment finished, project_name='{}', model_name='{}', version='{}',"
                          " is_fallback={}, cost {}s".format(project_name, model_name, version, is_fallback,
                                                             time.perf_counter()-st))
                    return
                elif status == self.INDEX_STATUS_FAILED:
                    print("embedding deployment failed, project_name='{}', model_name='{}', version='{}',"
                          " is_fallback={}, resp={}, cost {}s".format(project_name, model_name, version, is_fallback,
                                                                      query_ret.toJSON(), time.perf_counter() - st))
                    raise RuntimeError("failed to deploy embedding: {}".format(status))
                time.sleep(interval)
                elapsed += interval
                if wait_time is not None and datetime.timedelta(seconds=elapsed) >= wait_time:
                    msg = "waiting embedding deployment time out({}), project_name='{}', model_name='{}', " \
                          "version='{}', is_fallback={}, resp={}".format(wait_time, project_name, model_name,
                                                                         version, is_fallback, query_ret.toJSON())
                    print(msg)
                    raise RuntimeError(msg)

                error_times = 0
                print("embedding(project_name='{}', model_name='{}', version='{}', is_fallback={}) deploying: {},"
                      " elapsed {}s".format(project_name, model_name, version, is_fallback, status, elapsed))
            except Exception as e:
                error_times += 1
                if error_times >= max_retry:
                    raise e
                sleep_time = min(2 ** (error_times - 1), 60)
                print("{}/{} try of querying embedding deployment failed: {}, will retry after {}s"
                      .format(error_times, max_retry, e, sleep_time))
                time.sleep(sleep_time)
