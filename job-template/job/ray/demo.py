import ray,os,time


@ray.remote
def fun1(index):
    # 这里是耗时的任务，函数内不能引用全局变量，只能使用函数内的局部变量。
    print(index)
    time.sleep(1)
    return 'back_data'


def main():
    tasks = []
    all_data=range(100)  # 假设要处理的所有数据
    for index in all_data:
        tasks.append(fun1.remote(index))  # 建立远程函数
    result = ray.get(tasks)  # 获取任务结果


if __name__ == "__main__":

    head_service_ip = os.getenv('RAY_HOST', '')
    if head_service_ip:
        # 集群模式
        head_host = head_service_ip + ".pipeline" + ":10001"
        ray.util.connect(head_host)
    else:
        # 本地模式
        ray.init()

    main()