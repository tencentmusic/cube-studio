# coding=utf-8
# @Time     : 2021/4/12 21:08
# @Auther   : lionpeng@tencent.com

from job.pkgs.context import JobComponentRunner
from job.pkgs.httpclients.tesla_client import TeslaClient


class TeslaJobHandler(JobComponentRunner):
    def job_func(self, jc_entry):
        flow_id = jc_entry.job.get('flow_id')
        if flow_id is None:
            raise RuntimeError("'flow_id' must be specified")
        flow_params = jc_entry.job.get('flow_params')
        wait_time = jc_entry.job.get('wait_time')
        client = TeslaClient(jc_entry.creator)
        client.start_and_wait(flow_id, flow_params, wait_time)


if __name__ == '__main__':
    handler = TeslaJobHandler("Tesla Job Template")
    handler.run()
