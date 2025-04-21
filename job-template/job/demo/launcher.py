
import argparse
import json
import os
import shutil
import redis,pickle
import pandas
import pysnooper
# 读取平台，任务流，任务，任务实例的相关信息

creator = os.getenv('KFJ_CREATOR', 'admin')                  # 任务流的创建者
runner = os.getenv('KFJ_RUNNER', 'admin')                    # 任务流的运行者
pipeline_name = os.getenv('KFJ_PIPELINE_NAME','test')        # 任务流的名称，或id
task_name = os.getenv('KFJ_TASK_NAME','test')                # 任务的名称或id
cpu = os.getenv('KFJ_TASK_RESOURCE_CPU','2')                 # 申请的cpu资源或内存资源
gpu = os.getenv('KFJ_TASK_RESOURCE_GPU','0')                 # 申请的gpu资源
host_ip = os.getenv('K8S_HOST_IP','xx.xx.xx.xx')             # 任务调度的主机ip
run_id = os.getenv('KFJ_RUN_ID','xx')                        # 任务流运行实例id


# 你想要实现的计算逻辑，通过文件输入输出
@pysnooper.snoop()
def do_something(args):
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    # 你的任务逻辑，从目录读取输入，进行计算，保存输出
    input = pandas.read_csv(input_file_path)        # 读取输入数据
    result = input*3                                  # 根据输入数据和输入参数，做任何你想要的逻辑
    result.to_csv(output_file_path, index=False)    # 保存结果数据

# 你想要实现的计算逻辑，通过缓存输入输出
@pysnooper.snoop()
def do_something_with_cache():
    cache = redis.Redis.from_url(os.getenv('KFJ_CACHE_URL', ''))  # 缓存地址，可以在上下有传递数据
    # 你的任务逻辑，从目录读取输入，进行计算，保存输出
    input = cache.hget(run_id, "output")
    if not input:
        raise ValueError("上游输出不存在")
    input = pickle.loads(input)
    result = input*3                                      # 根据输入数据和输入参数，做任何你想要的逻辑
    cache.hset(run_id, "output", pickle.dumps(result))    # 保存结果数据


# 保存任务指标数据
@pysnooper.snoop()
def save_metrics():

    # 只有分布式存储中的数据才可以被读取，所以先复制可视化内容到个人分布式存储目录
    os.makedirs(f'/mnt/{creator}/pipeline/example/visualization/',exist_ok=True)
    shutil.copy('metric.csv',f'/mnt/{creator}/pipeline/example/visualization/metric.csv')
    os.makedirs(f'/mnt/{creator}/pipeline/example/ml/',exist_ok=True)
    shutil.copy('test_roc.png',f'/mnt/{creator}/pipeline/example/ml/test_roc.png')
    # 配置可视化参数
    metrics=[
        {
            "metric_type":"image",
            "describe":"这里添加图片的描述",
            "image": f'/mnt/{creator}/pipeline/example/ml/test_roc.png'
        },
        {
            "metric_type": "text",
            "describe": "这里添加文本的描述",
            "text": "auc=0.8"
        },
        {
            "metric_type": "table",
            "describe": "这里添加表格的描述",
            "file_path": f"/mnt/{creator}/pipeline/example/visualization/metric.csv"
        },
        {
            "metric_type": "echart-parallel",
            "describe": "csv 并行线可视化",
            "file_path": f"/mnt/{creator}/pipeline/example/visualization/metric.csv"
        }
    ]
    json.dump(metrics, open('/metric.json', mode='w'))


if __name__ == "__main__":
    # 接收用户的输入参数
    arg_parser = argparse.ArgumentParser("demo launcher")
    arg_parser.add_argument('--input_file_path', type=str, help="输入文件地址", default='data.csv')
    arg_parser.add_argument('--output_file_path', type=str, help="输出文件地址", default='data-result.csv')
    arg_parser.add_argument('--kwargs', type=str, help="其他参数", default='value')

    args = arg_parser.parse_args()
    try:
        args.kwargs = json.loads(args.kwargs)
    except Exception as e:
        print(e)
    if args.input_file_path:
        do_something(args=args)
    else:
        do_something_with_cache()

    # 保存计算指标，在界面可视化
    save_metrics()

# python launcher.py
