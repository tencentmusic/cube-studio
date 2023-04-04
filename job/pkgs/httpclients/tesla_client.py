# coding=utf-8
# @Time     : 2021/4/12 15:17
# @Auther   : lionpeng@tencent.com

import requests
import json
import traceback
import datetime
import time
from ..exceptions.tesla_exceptions import *


TESLA_BASE_URL = "http://taijiapi.oa.com"
TESLA_START_FLOW_API_URI = TESLA_BASE_URL + "/api/flow/runParamFlow.do"
TESLA_QUERY_FLOW_API_URI = TESLA_BASE_URL + "/api/flow/getParamFlow.do"


class TeslaClient(object):
    def __init__(self, auth_user):
        self.auth_user = auth_user

    def start_flow(self, flow_id, flow_params=None):
        data = {
            "flowId": flow_id,
            "authToken": self.auth_user
        }

        if not isinstance(flow_params, dict) or not flow_params:
            flow_params = {}
        data['paramPackage'] = json.dumps(flow_params)

        api_url = TESLA_START_FLOW_API_URI
        try:
            resp = self._request('post', api_url, data=data)
            if resp.status_code // 100 > 2:
                print("start tesla work flow http failed, api_url='{}', data={}, resp=[{}, {}, {}]"
                      .format(api_url, data, resp.status_code, resp.reason, resp.text))
                raise StartFLowException("start tesla work flow http error: [{}, {}, {}]"
                                         .format(resp.status_code, resp.reason, resp.text))
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', data={}, resp=[{}, {}, {}]"
                      .format(api_url, data, resp.status_code, resp.reason, resp.text))
                raise e1
            if not resp_body.get('success'):
                print("start tesla work flow failed, api_url='{}', data={}, resp_body={}"
                      .format(api_url, data, resp_body))
                raise StartFLowException("start tesla work flow error: {}".format(resp_body))
            return resp_body['data']
        except Exception as e:
            print("start tesla work flow error, api_url='{}', data={}: {}\n{}"
                  .format(api_url, data, e, traceback.format_exc()))
            raise e

    def query_flow(self, model_flow_id, uuid):
        data = {
            "flowId": model_flow_id,
            "jobFlowId": uuid
        }
        api_url = TESLA_QUERY_FLOW_API_URI
        try:
            resp = self._request('post', api_url, data=data)
            if resp.status_code // 100 > 2:
                print("query tesla work flow http failed, api_url='{}', data={}, resp=[{}, {}, {}]"
                      .format(api_url, data, resp.status_code, resp.reason, resp.text))
                raise QueryFlowException("query tesla work flow http error: [{}, {}, {}]"
                                         .format(resp.status_code, resp.reason, resp.text))
            try:
                resp_body = json.loads(resp.text)
            except Exception as e1:
                print("load response as json failed, api_url='{}', data={}, resp=[{}, {}, {}]"
                      .format(api_url, data, resp.status_code, resp.reason, resp.text))
                raise e1
            if not resp_body.get('success'):
                print("query tesla work flow failed, api_url='{}', data={}, resp_body={}"
                      .format(api_url, data, resp_body))
                raise StartFLowException("query tesla work flow error: {}".format(resp_body))
            return resp_body['data']
        except Exception as e:
            print("query tesla work flow error, api_url='{}', data={}: {}\n{}"
                  .format(api_url, data, e, traceback.format_exc()))
            raise e

    def start_and_wait(self, flow_id, flow_params=None, wait_time=None):
        start_ret = self.start_flow(flow_id, flow_params)
        model_flow_id = start_ret['modelFlowId']
        uuid = start_ret['uuid']
        retries = 0
        max_retries = 15
        st = time.perf_counter()
        if wait_time:
            from ..utils import parse_timedelta
            wait_time = parse_timedelta(wait_time)
        print("begin waiting tesla work flow '{}' to complete, uuid='{}'".format(flow_id, uuid))
        while True:
            try:
                query_ret = self.query_flow(model_flow_id, uuid)
                status = query_ret['status']
                if status in ['unscheduled', 'failed', 'killed']:
                    print("tesla work flow {} failed with status '{}', flow_params={}, cost {}s, status detail: {}"
                          .format(flow_id, status, flow_params, time.perf_counter()-st, query_ret))
                    raise FlowFailedException("tesla work flow {} failed with status '{}'".format(flow_id, status))
                elif status == 'finished':
                    print("tesla work flow {} finished, uuid='{}', cost {}s".format(flow_id, uuid,
                                                                                    time.perf_counter()-st))
                    return
                elapsed = time.perf_counter() - st
                print("waiting tesla work flow {} to complete, uuid='{}', current status '{}', elapsed {}s"
                      .format(flow_id, uuid, status, elapsed))
                if wait_time is not None:
                    if datetime.timedelta(seconds=elapsed) >= wait_time:
                        print("tesla work flow {} timeout({}), uuid='{}'".format(flow_id, wait_time, uuid))
                        raise FLowTimeoutException("tesla work flow {} time out".format(flow_id))
                time.sleep(5)
            except Exception as e:
                if isinstance(e, (FlowFailedException, FLowTimeoutException)):
                    raise e
                retries += 1
                if retries < max_retries:
                    print("failed to query tesla work flow, will retry after 10s ({}/{})"
                          .format(retries, max_retries))
                    time.sleep(10)
                    continue
                print("start and wait tesla work flow error: {}\n{}".format(e, traceback.format_exc()))
                raise e

    @staticmethod
    def _request(method, url, params=None, data=None):
        return requests.request(method, url, params=params, data=data)
