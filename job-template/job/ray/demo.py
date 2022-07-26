import ray,os,time


@ray.remote
def fun1(arg):
    # 这里是耗时的任务，函数内不能引用全局变量，只能使用函数内的局部变量。
    print(arg)
    time.sleep(1)
    return 'back_data'


def main():
    tasks=[]
    tasks_args = range(100)
    for arg in tasks_args:
        tasks.append(fun1.remote(arg))  # 建立远程函数
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