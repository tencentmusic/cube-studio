pytorch在k8s之上的分布式训练

使用该模板能够帮助你在k8s自动创建一个pytorch分布式训练的集群。但是前提是需要你按照pytorch官方的方案先将代码编写为分布式形式。

# 单机版示例

https://github.com/pytorch/examples/blob/master/mnist/main.py

# 分布式版示例

https://github.com/kubeflow/pytorch-operator/blob/master/examples/mnist/mnist.py

# 分布式原理

![图片 1](https://user-images.githubusercontent.com/20157705/169199294-257074bb-bea2-4077-bd45-b6e223694879.png)

### 基本原则
每个进程的rank是不能一样的，进程总数目是为WORLD_SIZE，master只能是rank=0

### 主要变更
```python
分布式集群的每个pod，都会提供如下环境变量
NCCL_DEBUG=INFO
NCCL_IB_DISABLE="1"
MASTER_PORT="23456"
NCCL_SOCKET_IFNAME=eth0
MASTER_ADDR=pytorchjob-xxx-master-0
WORLD_SIZE=3    # 一共多少个worker
RANK=0   # 当前是第几个worker

注意：
master：RANK=0，worker-0：RANK=1，worker-1：RANK=2

# 初始化集群信息
if int(os.environ.get('WORLD_SIZE', 1))>1:
    # 要不专门配置init_method RANK或者WORLD_SIZE  系统会自动识别
    dist.init_process_group(backend=args.backend,init_method=None)

# 被DDP封装的model的参数的grad才会进行all reduce
if is_distributed():
    Distributor = nn.parallel.DistributedDataParallel if use_cuda else nn.parallel.DistributedDataParallelCPU   
    model = Distributor(model)

# 需要DistributedSampler作为实例传递给DataLoader来配合DDP使用，这样数据集的样本会为每个进程划分，每个进程读取各自的样本。
train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)  

# 分布式下set_epoch
train_sampler.set_epoch(epoch)
```

# 启动方式
### 直接python启动your_start.py
例如上面的mnist代码
### torch.distributed.launch启动
```bash
python -m torch.distributed.launch --nproc_per_node=每个worker的卡数量 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py --自己脚本的其他参数
```
torch.distributed.launch会向你的脚本传递--local_rank参数，同时会透传train.py后面的参数
你的train.py脚本

```python
if __name__ == "__main__":
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
	
world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])
dist.init_process_group('nccl')
```
# gpu利用率优化
其中gpu由于是整卡占用，需要调整任务的部分参数和代码，提高gpu显存占用率和gpu使用率
## 平台监控
通过监控按钮，可以进入查看任务运行的资源使用率，对于资源使用超标，可以手动配置增加资源。
![tapd_20424693_1635928300_4](https://user-images.githubusercontent.com/20157705/169199941-9324b23f-70a1-4839-b2a9-cae513d85ad8.png)
自己监控利用率
```bash
watch nvidia-smi
或者
pip install gpustat
watch --color -n1 gpustat -cpu
```
## gpu利用率低的原因
![tapd_20424693_1635939945_56](https://user-images.githubusercontent.com/20157705/169200232-ec11dc14-e1e5-48a7-a261-bbfbab9a6c0a.png)

核心：cpu操作慢，进而阻塞了gpu的计算

可能的原因：数据加载/网络等待/数据预处理/模型保存/loss 计算/评估指标计算/日志打印/指标上报/进度上报

## gpu利用率优化

### **1、数据加载相关**
 - 1、存储计算不在同一个城市：数据导入到集群存储
 - 2、磁盘io性能太差：对于临时数据可以将内存映射为磁盘
 - 3、小文件太多，频繁io：合并为大文件处理
 - 4、未启用多进程并行读取数据：pytorch提高num_workers，tf配置num_parallel_calls/num_parallel_reads
 - 5、未启用提前加载机制来实现 CPU 和 GPU 的并行：pytorch配置prefetch_factor，tf配置Dataset.prefetch()
 - 6、未设置共享内存 pin_memory：设置为true
 - 7、每次送入gpu的_size太少：模型固定后，调整 batch_size，尽量增大显存的利用率。然后再调节num_workers提高gpu利用率

### **2、数据预处理相关**
 - 1、数据处理和训练耦合在一起：将数据处理和训练分成两个task，训练中需要的配置之类的全部提前加载到内存，让gpu只做训练任务。或者使用Nvidia DALI，在gpu中做数据处理

### **3、频繁io操作**
 - 1、模型保存太频繁：减少保存模型(checkpoint)的频率
 - 2、tensorboard文件保存太频繁：xxxx
 - 3、日志打印太频繁，频繁cpu/gpu切换：不要打印训练迭代中个人日志

# 多进程共享gpu
通过多进程共享单机的方式，提高gpu的利用率概念图。

![图片 1](https://user-images.githubusercontent.com/20157705/169200585-7159254d-429c-45e8-b564-2d5ea7f06df3.png)



##  添加多进程共享gpu卡的启动方式

#### shell方式添加start端

可以添加启动start.sh，启动3个进程。每个进程在原有基础上添加`--process_index xx --process_num xx` 参数。并放在在后端运行，并在最后wait所有后端程序。

```
python3 /mnt/pengluan/mytask.py --lr xx ... --process_index 0 --process_num 3 > /process0.file 2>&1 &

python3 /mnt/pengluan/mytask.py --lr xx ... --process_index 1 --process_num 3 > /process1.file 2>&1 &

python3 /mnt/pengluan/mytask.py --lr xx ... --process_index 2 --process_num 3 > /process2.file 2>&1 &

wait
```
#### python方式添加start端（透传上层参数）

添加一个start.py，通过start.py启动多个任务进程
```
import json
import argparse
import subprocess
import sys

if __name__ == '__main__':
    # 以下参数列表只是示例，实际使用时请按需自己增删改
    arg_parser = argparse.ArgumentParser("多进程启动")
    process_num=3
    python_path="/mnt/pengluan/mytask.py"
    commands = [["/usr/bin/python3",python_path,"--process_index",str(process_index),"--process_num",str(process_num)]+sys.argv[1:] for process_index in range(process_num)]
    print(commands)
    all_process = [subprocess.Popen(command) for command in commands]
    all_returncode = [process.wait() for process in all_process]

```
