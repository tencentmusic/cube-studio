# ray-sklearn 模板
描述：基于ray的分布式能力，实现sklearn机器学习模型的分布式训练。  

镜像：ccr.ccs.tencentyun.com/cube-studio/sklearn_estimator:v1  

环境变量：  
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=2
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```

启动参数：  
```bash
train_csv_file_path: 训练集csv，|分割符，首行是列名
predict_csv_file_path: 预测数据集csv，格式和训练集一致，默认为空，需要predict时填
label_name: label的列名，必填
model_name: 训练用到的模型名称，如LogisticRegression，必填。常用的都支持，要加联系管理员
model_args_dict: 模型参数，json格式，默认为空
model_file_path: 模型文件保存文件名，必填
predict_result_path: 预测结果保存文件名，默认为空，需要predict时填
worker_num: ray worker数量
```

