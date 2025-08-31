import json, datetime, time
import requests
import pysnooper


class Prometheus():

    def __init__(self, host=''):
        #  '/api/v1/query_range'    查看范围数据
        #  '/api/v1/query'    瞬时数据查询
        self.host = host
        self.query_path = 'http://%s/api/v1/query' % self.host
        self.query_range_path = 'http://%s/api/v1/query_range' % self.host

    # @pysnooper.snoop()
    def get_istio_service_metric(self, namespace):
        service_metric = {
            "qps": {},
            "gpu": {},
            "memory": {},
            "cpu": {}
        }
        # qps请求
        mem_expr = 'sum by (destination_workload,response_code) (irate(istio_requests_total{destination_service_namespace="%s"}[1m]))' % (namespace,)
        # print(mem_expr)
        params = {
            'query': mem_expr,
            'start': int(time.time())-300,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)

        try:
            res = requests.get(url=self.query_range_path, params=params)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    for service in metrics:
                        service_name = service['metric']['destination_workload']
                        if service_name not in service_metric['qps']:
                            service_metric['qps'][service_name] = {}
                        service_metric["qps"][service_name] = service['values']

        except Exception as e:
            print(e)

        # 内存
        mem_expr = 'sum by (pod) (container_memory_working_set_bytes{job="kubelet", image!="",container_name!="POD",namespace="%s"})' % (namespace,)
        # print(mem_expr)
        params = {
            'query': mem_expr,
            'start': int(time.time()) - 300,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)

        try:
            res = requests.get(url=self.query_range_path, params=params)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    for pod in metrics:
                        pod_name = pod['metric']['pod']
                        if pod_name not in service_metric['memory']:
                            service_metric[pod_name] = {}
                        service_metric['memory'][pod_name] = pod['values']

        except Exception as e:
            print(e)

        # cpu获取
        cpu_expr = "sum by (pod) (rate(container_cpu_usage_seconds_total{namespace='%s',container!='POD'}[1m]))" % (namespace)

        params = {
            'query': cpu_expr,
            'start': int(time.time()) - 300,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:

            res = requests.get(url=self.query_range_path, params=params)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    for pod in metrics:
                        pod_name = pod['metric']['pod']
                        if pod_name not in service_metric['cpu']:
                            service_metric[pod_name] = {}
                        service_metric['cpu'][pod_name] = pod['values']

        except Exception as e:
            print(e)

        gpu_expr = "avg by (pod) (DCGM_FI_DEV_GPU_UTIL{namespace='%s'})" % (namespace)

        params = {
            'query': gpu_expr,
            'start': (datetime.datetime.now() - datetime.timedelta(days=1) - datetime.timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'end': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:

            res = requests.get(url=self.query_range_path, params=params)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    # print(metrics)
                    for pod in metrics:
                        pod_name = pod['metric']['pod']
                        if pod_name not in service_metric['gpu']:
                            service_metric['gpu'][pod_name] = {}
                        service_metric['gpu'][pod_name] = pod['values']

        except Exception as e:
            print(e)

        return service_metric



    # 获取当前pod利用率
    # @pysnooper.snoop()
    def get_resource_metric(self):
        max_cpu = 0
        max_mem = 0
        ave_gpu = 0
        pod_metric = {}
        # 这个pod  30分钟内的最大值
        mem_expr = "sum by (pod) (container_memory_working_set_bytes{container!='POD', container!=''})"
        # print(mem_expr)
        params = {
            'query': mem_expr,
            'start': int(time.time()) - 300,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)

        try:
            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    for pod in metrics:
                        if pod['metric']:
                            pod_name = pod['metric']['pod']
                            values = max([float(x[1]) for x in pod['values']])
                            if pod_name not in pod_metric:
                                pod_metric[pod_name] = {}
                            pod_metric[pod_name]['memory'] = round(values / 1024 / 1024 / 1024, 2)

        except Exception as e:
            print(e)

        cpu_expr = "sum by (pod) (rate(container_cpu_usage_seconds_total{container!='POD'}[1m]))"

        params = {
            'query': cpu_expr,
            'start': int(time.time()) - 300,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:

            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    for pod in metrics:
                        if pod['metric']:
                            pod_name = pod['metric']['pod']
                            values = [float(x[1]) for x in pod['values']]
                            # values = round(sum(values) / len(values), 2)
                            values = round(max(values), 2)
                            if pod_name not in pod_metric:
                                pod_metric[pod_name] = {}
                            pod_metric[pod_name]['cpu'] = values

        except Exception as e:
            print(e)

        # gpu的资源利用率
        gpu_expr = "avg by (pod) (DCGM_FI_DEV_GPU_UTIL)"

        params = {
            'query': gpu_expr,
            'start': int(time.time()) - 300,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:

            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    # print(metrics)
                    for pod in metrics:
                        if pod['metric']:
                            pod_name = pod['metric']['pod']
                            values = [float(x[1]) for x in pod['values']]
                            # values = round(sum(values)/len(values),2)
                            values = round(max(values), 2)
                            if pod_name not in pod_metric:
                                pod_metric[pod_name] = {}
                            pod_metric[pod_name]['gpu'] = values / 100

        except Exception as e:
            print(e)

        return pod_metric

    # @pysnooper.snoop()
    def get_namespace_resource_metric(self, namespace):
        max_cpu = 0
        max_mem = 0
        ave_gpu = 0
        pod_metric = {}
        # 这个pod  30分钟内的最大值
        mem_expr = "sum by (pod) (container_memory_working_set_bytes{namespace='%s',container!='POD', container!=''})" % (namespace,)
        # print(mem_expr)
        params = {
            'query': mem_expr,
            'start': int(time.time()) - 60*60*24,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)

        try:
            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    for pod in metrics:
                        pod_name = pod['metric']['pod']
                        values = max([float(x[1]) for x in pod['values']])
                        if pod_name not in pod_metric:
                            pod_metric[pod_name] = {}
                        pod_metric[pod_name]['memory'] = round(values / 1024 / 1024 / 1024, 2)

        except Exception as e:
            print(e)

        # 获取cpu的利用率
        cpu_expr = "sum by (pod) (rate(container_cpu_usage_seconds_total{namespace='%s',container!='POD'}[1m]))" % (namespace)

        params = {
            'query': cpu_expr,
            'start': int(time.time()) - 60*60*24,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:

            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    for pod in metrics:
                        pod_name = pod['metric']['pod']
                        values = [float(x[1]) for x in pod['values']]
                        # values = round(sum(values) / len(values), 2)
                        values = round(max(values), 2)
                        if pod_name not in pod_metric:
                            pod_metric[pod_name] = {}
                        pod_metric[pod_name]['cpu'] = values

        except Exception as e:
            print(e)


        # 获取gpu的利用率
        gpu_expr = "avg by (pod) (DCGM_FI_DEV_GPU_UTIL{namespace='%s'})" % (namespace)

        params = {
            'query': gpu_expr,
            'start': int(time.time()) - 60*60*24,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:

            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    # print(metrics)
                    for pod in metrics:
                        pod_name = pod['metric']['pod']
                        values = [float(x[1]) for x in pod['values']]
                        # values = round(sum(values)/len(values),2)
                        values = round(max(values), 2)
                        if pod_name not in pod_metric:
                            pod_metric[pod_name] = {}
                        pod_metric[pod_name]['gpu'] = values / 100

        except Exception as e:
            print(e)



        return pod_metric

        # @pysnooper.snoop()

    def get_pod_resource_metric(self, pod_name, namespace):
        max_cpu = 0
        max_mem = 0
        ave_gpu = 0

        # 这个pod  30分钟内的最大值
        mem_expr = "sum by (pod) (container_memory_working_set_bytes{namespace='%s', pod=~'%s.*',container!='POD', container!=''})"%(namespace,pod_name)
        # print(mem_expr)
        params = {
            'query': mem_expr,
            'start': int(time.time()) - 60*60*24,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)

        try:
            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    metrics = metrics[0]['values']
                    for metric in metrics:
                        if int(metric[1]) > max_mem:
                            max_mem = int(metric[1]) / 1024 / 1024 / 1024

        except Exception as e:
            print(e)

        cpu_expr = "sum by (pod) (rate(container_cpu_usage_seconds_total{namespace='%s',pod=~'%s.*',container!='POD'}[1m]))" % (namespace, pod_name)

        params = {
            'query': cpu_expr,
            'start': int(time.time()) - 60*60*24,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:

            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    metrics = metrics[0]['values']
                    for metric in metrics:
                        if float(metric[1]) > max_cpu:
                            max_cpu = float(metric[1])
        except Exception as e:
            print(e)

        gpu_expr = "avg by (pod) (DCGM_FI_DEV_GPU_UTIL{namespace='%s',pod=~'%s.*'})" % (namespace, pod_name)

        params = {
            'query': gpu_expr,
            'start': int(time.time()) - 60*60*24,
            'end': int(time.time()),
            'step': "1m",  # 运行小于1分钟的，将不会被采集到
            # 'timeout':"30s"
        }
        # print(params)
        try:
            res = requests.get(url=self.query_range_path, params=params,timeout=5)
            metrics = json.loads(res.content.decode('utf8', 'ignore'))
            if metrics['status'] == 'success':
                metrics = metrics['data']['result']
                if metrics:
                    metrics = metrics[0]['values']
                    all_util = [float(metric[1]) for metric in metrics]
                    ave_gpu = sum(all_util) / len(all_util) / 100
        except Exception as e:
            print(e)

        return {"cpu": round(max_cpu, 2), "memory": round(max_mem, 2), 'gpu': round(ave_gpu, 2)}

    # todo 获取机器的负载补充完整
    # @pysnooper.snoop()
    def get_machine_metric(self):
        # 这个pod  30分钟内的最大值
        metrics = {
            "pod_num": "sum(kubelet_running_pod_count)by (node)",
            "request_memory": "",
            "request_cpu": "",
            "request_gpu": "",
            "used_memory": "",
            "used_cpu": "",
            "used_gpu": "",
        }
        back = {}
        for metric_name in metrics:
            # print(mem_expr)
            params = {
                'query': metrics[metric_name],
                'timeout': "30s"
            }
            # print(params)
            back[metric_name] = {}

            try:
                res = requests.get(url=self.query_path, params=params)
                metrics = json.loads(res.content.decode('utf8', 'ignore'))
                if metrics['status'] == 'success':
                    metrics = metrics['data']['result']
                    if metrics:
                        for metric in metrics:
                            node = metric['metric']['node']
                            if ':' in node:
                                node = node[:node.index(':')]
                            value = metric['value'][1]
                            back[metric_name][node] = int(value)


            except Exception as e:
                print(e)

        return back


if __name__ == "__main__":
    prometheus = Prometheus('10.101.142.16:8081')
    result = prometheus.get_istio_service_metric('service')
    # print(result)
