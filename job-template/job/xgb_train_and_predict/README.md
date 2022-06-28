# xgboost 模板
描述：单机xgb训练，支持训练预测。

镜像：ccr.ccs.tencentyun.com/cube-studio/xgb_train_and_predict:v1  

环境变量：  
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=2
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```

启动参数：  
```bash
sep： 分隔符
classifier_or_regressor： 分类还是回归
params： xgb参数, json格式，透传给xgboost类构建函数
train_csv_file_path： 训练集csv路径，首行是header，首列是label。为空则不做训练，尝试从model_load_path加载模型。
model_load_path： 模型加载路径。为空则不加载。
predict_csv_file_path： 预测数据集csv路径，格式和训练集一致，顺序保持一致，没有label列。为空则不做predict。
predict_result_path： 预测结果保存路径，为空则不做predict。
model_save_path： 模型文件保存路径。为空则不保存模型。
eval_result_path： 模型评估报告保存路径。默认为空，想看模型评估报告可填。
```
