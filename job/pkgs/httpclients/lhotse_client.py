# coding=utf-8
# @Time     : 2021/4/15 20:13
# @Auther   : lionpeng@tencent.com

import requests
import json
import traceback
import datetime
import time
from dateutil import relativedelta
from ..exceptions.lhotse_exceptions import *
from ..tdw.tdw_tauth_authentication import TdwTauthAuthentication

API_BASE_URL = "http://tdwopen.oa.com"

DATE_FMT = '%Y-%m-%d %H:%M:%S'



class LhotseClient(object):
    def __init__(self, tdw_sec_file):
        self.tta = TdwTauthAuthentication(tdw_sec_file, "Common-Scheduler")
        self.req_headers = {"Version": "2"}

    def check_task(self, task_id=None, task_name=None):
        api_url = API_BASE_URL + "/Uscheduler/LhotseCheck"
        if task_id is not None:
            params = {"taskId": task_id}
        elif task_name is not None:
            params = {"taskName": task_name}
        else:
            print("WARN: neither 'task_id' nor 'task_name' were specified")
            return None
        try:
            resp = self._request('get', api_url, params=params)
            if resp.status_code // 100 > 2:
                msg = "check lhotse task '{}'/'{}' http error, api_url='{}', response=[{}, {}, {}]"\
                    .format(task_id, task_name, api_url, resp.status_code, resp.reason, resp.text)
                raise CheckTaskException(msg)
            resp_body = self.__parse_response_content(resp)
            if not isinstance(resp_body, list) or not resp_body:
                msg = "invalid response when check lhotse task '{}'/'{}', api_url='{}', resp_body={}"\
                    .format(task_id, task_name, api_url, resp_body)
                raise CheckTaskException(msg)
            resp_body = resp_body[0]
            if resp_body.get('state') != 'success':
                msg = "check lhotse task '{}'/'{}' server error, api_url='{}', resp_body={}"\
                    .format(task_id, task_name, api_url, resp_body)
                raise CheckTaskException(msg)
            if resp_body.get('status') != 'Y':
                msg = "lhotse task '{}'/'{}' is abnormal, detail: {}" .format(task_id, task_name, resp_body)
                raise TaskAbnormalException(msg)
        except Exception as e:
            print("check lhotse task '{}'/'{}' error, api_url='{}', params={}: {}\n{}"
                  .format(task_id, task_name, api_url, params, e, traceback.format_exc()))
            raise e

    def query_task_config(self, task_id):
        api_url = API_BASE_URL + "/Uscheduler/QueryTask"
        params = {"taskId": task_id}
        try:
            resp = self._request('get', api_url, params=params)
            if resp.status_code // 100 > 2:
                msg = "query config of lhotse task '{}' http error, api_url='{}', response=[{}, {}, {}]"\
                    .format(task_id, api_url, resp.status_code, resp.reason, resp.text)
                raise QueryTaskException(msg)
            resp_body = self.__parse_response_content(resp)
            if not isinstance(resp_body, list) or not resp_body or not resp_body[0]:
                msg = "invalid response when query lhotse task '{}', api_url='{}', resp_body={}"\
                    .format(task_id, api_url, resp_body)
                raise QueryTaskException(msg)
            return resp_body[0]
        except Exception as e:
            print("query config of lhotse task '{}' error, api_url='{}', params={}: {}\n{}"
                  .format(task_id, api_url, params, e, traceback.format_exc()))
            raise e

    def query_task_instance_status(self, task_id, start_time=None, end_time=None, page_size=None):
        api_url = API_BASE_URL + "/Uscheduler/QueryTaskRun"
        params = {"task_id": task_id}

        if end_time is None:
            end_time = datetime.datetime.now()
        elif isinstance(end_time, str):
            end_time = datetime.datetime.strptime(end_time, DATE_FMT)
        elif not isinstance(end_time, datetime.datetime):
            raise RuntimeError("'end_time' should be a None/'{}' formated str/datetime, got '{}': {}"
                               .format(DATE_FMT, type(end_time), end_time))

        if start_time is None:
            start_time = end_time - datetime.timedelta(days=1)
        elif isinstance(start_time, str):
            start_time = datetime.datetime.strptime(start_time, DATE_FMT)
        elif not isinstance(start_time, datetime.datetime):
            raise RuntimeError("'start_time' should be a None/'{}' formated str/datetime, got '{}': {}"
                               .format(DATE_FMT, type(start_time), start_time))

        if start_time > end_time:
            raise RuntimeError("start time {} must not greater than end time {}".format(start_time, end_time))
        if not page_size:
            page_size = 1
        params['startTime'] = start_time.strftime(DATE_FMT)
        params['endTime'] = end_time.strftime(DATE_FMT)
        params['pageSize'] = page_size
        try:
            resp = self._request('get', api_url, params=params)
            if resp.status_code // 100 > 2:
                msg = "query instances of lhotse task '{}' http error, api_url='{}', response=[{}, {}, {}]" \
                    .format(task_id, api_url, resp.status_code, resp.reason, resp.text)
                raise QueryTaskInstanceException(msg)
            resp_body = self.__parse_response_content(resp)
            if not isinstance(resp_body, list) or not resp_body or not resp_body[0]:
                msg = "invalid response when query instances of lhotse task '{}', api_url='{}', resp_body={}"\
                    .format(task_id, api_url, resp_body)
                raise QueryTaskInstanceException(msg)
            resp_body = resp_body[0]
            if resp_body.get('state') != 'success':
                msg = "query instances of lhotse task '{}' server error, api_url='{}', resp_body={}" \
                    .format(task_id, api_url, resp_body)
                raise QueryTaskInstanceException(msg)
            instances = resp_body.get('desc', '[]')
            instances = json.loads(instances)
            return sorted(instances, key=lambda x: x['cur_run_date'], reverse=True)
        except Exception as e:
            print("query instances of lhotse task '{}' error, api_url='{}', params={}: {}\n{}"
                  .format(task_id, api_url, params, e, traceback.format_exc()))
            raise e

    def wait_task_instances(self, task_id, start_time=None, end_time=None, wait_time=None):
        task_cfg = self.query_task_config(task_id)
        if task_cfg.get('status') != 'Y':
            msg = "lhotse task '{} is abnormal, detail: {}".format(task_id, task_cfg)
            raise TaskAbnormalException(msg)

        cycle_num = task_cfg.get('cycleNum', 0)
        cycle_unit = task_cfg.get('cycleUnit')
        is_cyclic = cycle_num > 0 and (cycle_unit in ['I', 'M', 'H', 'W', 'D'])
        print("lhotse task '{}' cycle_num='{}', cycle_unit='{}', is_cyclic={}"
              .format(task_id, cycle_num, cycle_unit, is_cyclic))

        if wait_time:
            from ..utils import parse_timedelta
            wait_time = parse_timedelta(wait_time)

        if is_cyclic:
            if cycle_unit == 'I':
                cycle_len = datetime.timedelta(minutes=cycle_num)
            elif cycle_unit == 'M':
                now = datetime.datetime.now()
                cycle_len = (now+relativedelta.relativedelta(months=cycle_num))-now
            elif cycle_unit == 'H':
                cycle_len = datetime.timedelta(hours=cycle_num)
            elif cycle_unit == 'W':
                cycle_len = datetime.timedelta(weeks=cycle_num)
            elif cycle_unit == 'D':
                cycle_len = datetime.timedelta(days=cycle_num)

            if end_time is None:
                if start_time is not None:
                    end_time = start_time
                else:
                    end_time = datetime.datetime.now()
                print("'end_time' is not set, default to {}".format(end_time))
            if isinstance(end_time, str):
                end_time = datetime.datetime.strptime(end_time, DATE_FMT)
            elif not isinstance(end_time, datetime.datetime):
                raise RuntimeError("'end_time' should be a None/'{}' formated str/datetime, got '{}': {}"
                                   .format(DATE_FMT, type(end_time), end_time))

            if start_time is None:
                start_time = end_time - cycle_len
                print("'start_time' is not set, auto set to {} according to cycle length".format(start_time))
            elif isinstance(start_time, str):
                start_time = datetime.datetime.strptime(start_time, DATE_FMT)
            elif not isinstance(start_time, datetime.datetime):
                raise RuntimeError("'start_time' should be a None/'{}' formated str/datetime, got '{}': {}"
                                   .format(DATE_FMT, type(start_time), start_time))

            span_secs = int((end_time-start_time).total_seconds())
            cycle_secs = int(cycle_len.total_seconds())
            cycle_count = max(span_secs // cycle_secs, 1)
            print("auto set 'page_size'={}".format(cycle_count))
        else:
            start_time = None
            end_time = None
            cycle_count = 1
            print("non-cyclic task, set 'start_time' and 'end_time' to None, 'page_size' to 1")

        st = time.perf_counter()
        retries = 0
        max_retries = 15
        print("begin waiting instances of lhotse task '{}' to finish, 'start_time'={}, 'end_time'={}, page_size={}"
              .format(task_id, start_time, end_time, cycle_count))
        while True:
            try:
                instances = self.query_task_instance_status(task_id, start_time, end_time, cycle_count)
                for ins in instances:
                    if not isinstance(ins, dict) or ins.get('state') not in ['-1', '0', '1', '2', '9']:
                        if ins.get('state') == '3':
                            tries = ins.get('tries', 0)
                            try_limit = ins.get('try_limit', 0)
                            if tries < try_limit:
                                print("WARN: {}/{} try of instance of lhotse task '{}' failed: {}"
                                      .format(tries, try_limit, task_id, ins))
                                continue
                        msg = "found failed instance of lhotse task '{}': {}".format(task_id, ins)
                        print(msg)
                        raise TaskInstanceFailedException(msg)
                if len(instances) >= cycle_count and all([i.get('state') == '2' for i in instances]):
                    print("all {} instances of lhotse task '{}' between {} and {} finished, cost {}s"
                          .format(len(instances), task_id, start_time, end_time, time.perf_counter()-st))
                    return
                ins_states = [i.get('state') for i in instances]
                elapsed = time.perf_counter() - st
                print("found {}/{} instances of lhotse task '{}' between {} and {}, states: {}, elapsed {}s"
                      .format(len(instances), cycle_count, task_id, start_time, end_time, ins_states, elapsed))
                if wait_time is not None:
                    if datetime.timedelta(seconds=elapsed) >= wait_time:
                        msg = "waiting instances of lhotse task '{}' time out({})".format(task_id, wait_time)
                        print(msg)
                        raise WaitTimeoutException(msg)
                time.sleep(10)
            except Exception as e:
                if isinstance(e, (TaskInstanceFailedException, WaitTimeoutException)):
                    raise e
                retries += 1
                if retries < max_retries:
                    print("failed to query instance of lhotse task '{}', will retry after 10s ({}/{})"
                          .format(task_id, retries, max_retries))
                    time.sleep(10)
                    continue
                print("wait instance of lhotse task '{}' error: {}\n{}".format(task_id, e, traceback.format_exc()))
                raise e

    def _request(self, method, url, params=None, data=None):
        return requests.request(method, url, params=params, json=data, headers=self._make_headers())

    def _make_headers(self):
        auth = self.tta.getAuthentication()
        self.req_headers.update(auth)
        return self.req_headers

    @classmethod
    def __parse_response_content(cls, resp: requests.Response):
        if not resp:
            raise RuntimeError("response can not be None")
        try:
            json_ctnt = json.loads(resp.text)
            return json_ctnt
        except Exception as e:
            raise RuntimeError("parse response content error: code={}, reason='{}', text='{}', headers={},"
                               " request headers={}"
                               .format(resp.status_code, resp.reason, resp.text, resp.headers,
                                       None if resp.request is None else resp.request.headers))
