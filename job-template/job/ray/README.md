# 安装包 
pip install ray

# 使用
原有代码

```
# import ray

def fun1(index):
    # 这里是耗时的任务
    return 'back_data'

def main():
    for index in [...]:
         fun1(index)    # 直接执行任务

if __name__=="__main__":
    main()
```

启用ray框架的代码
```
import ray

@ray.remote
def fun1(index):
    # 这里是耗时的任务，函数内不能引用全局变量，只能使用函数内的局部变量。
    return 'back_data'

def main():
    tasks=[]
    for index in [...]:
         tasks.append(fun1.remote(index))   # 建立远程函数
    result = ray.get(tasks)   #  获取任务结果

if __name__=="__main__":
    
    head_service_ip = os.getenv('RAY_HOST','')
    if head_service_ip:
        # 集群模式
        head_host = head_service_ip+".pipeline"+":10001"
        ray.util.connect(head_host)
    else:
        # 本地模式
        ray.init()
        
    main()
```

# 示例
demo.py