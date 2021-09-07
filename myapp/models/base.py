from flask_appbuilder import Model
from flask_babel import lazy_gettext as _
import re
from myapp.utils import core

# 定义model
class MyappModelBase():

    label_columns={
        "name": "名称",
        "name_url": "名称",
        "name_title": "名称(移动鼠标，查看描述)",
        'job_type': '任务类型',
        "project": "项目组",
        "project_url":"项目组",
        "namespace": "命名空间",
        "namespace_url": "命名空间",
        "describe": "描述",
        "describe_url": "描述",
        "password": '密码',
        "workdir":"工作目录",
        "images": "镜像",
        "repository": "仓库",
        "args": "启动参数",
        "args_html": "启动参数",
        "demo": "参数示例",
        "demo_html": "参数示例",
        "entrypoint": '启动命令',
        "dockerfile": 'Dockerfile',
        "gitpath": 'git地址',
        "env": "环境变量",
        "privileged": "特权模式",
        "accounts": "k8s账号",
        "images_url": "镜像",
        "hostAliases": "host",
        "overwrite_entrypoint": "覆盖原始启动命令",
        "command": "启动命令",
        "working_dir": "启动目录",
        "volume_mount": "挂载目录",
        "node_selector": "调度机器",
        "image_pull_policy": "镜像拉取策略",
        "resource_memory": "内存申请",
        "resource_cpu": "cpu申请",
        "resource_gpu": "gpu申请",
        "resource":"资源",
        "timeout": "超时中断",
        "retry": "重试次数",
        "outputs": "输出",
        "version": "版本",
        "model_name":"模型名称",
        "model_path":"模型地址",
        "embedding_file_path":"embedding文件地址",
        "is_fallback":"兜底版本",
        "check_service":"检查服务",
        "status": "状态",
        "status_url":"状态",
        "final_status":"最终状态",
        "pipeline": "任务流",
        "pipeline_url": "任务流",
        "run_id": "kfp运行id",
        "run_time": "kfp运行时间",
        "type": "类型",
        "reset": "重置",
        "user": "用户",
        "role": "角色",
        "dag_json": "流向图",
        "dag_json_html": "流向图",
        "username": "用户",
        "schedule_type": "调度类型",
        "cron_time": "调度周期",
        "global_args": "全局参数",
        "global_env": "全局环境变量",
        "parallelism": "任务并行数",
        "run_pipeline": "运行",
        "status_more": "状态详情",
        "status_more_html": "状态详情",
        "execution_date":"执行时间",
        "base_image":"基础镜像",
        "tag":"tag",
        "save":"保存",
        "history":"历史",
        "consecutive_build":"连续构建",

        "log": "日志",
        "pod": "容器",
        "ide_type": "IDE类型",
        "annotations": "注释",
        "annotations_html": "注释",
        "spec": "属性",
        "spec_html": "属性",
        "info_json": "通知",
        "info_json_html": "通知",
        "labels": "标签",
        "label": "标签",
        "labels_html": "标签",
        "label_url": "标签",
        "add_row_time": "添加时间",
        "experiment_id": "kfp实验id",
        "pipeline_file": "workflow yaml",
        "pipeline_file_html": "workflow yaml",
        "pipeline_argo_id": "kfp任务流id",
        "version_id": "kfp版本id",
        "job_template": "任务模板",
        "job_template_url": "任务模板",
        "alert_status": "监控状态",

        "alert_user": "通知用户",
        "experiment": "Experiment yaml",
        "experiment_html": "Experiment yaml",
        "train_model": "训练模型",
        "ip": "ip",
        "deploy_time": "部署时间",
        "host": "域名",
        "host_url": "域名",
        "deploy": "部署",
        "test_deploy": "测试部署",
        "prod_deploy": "生产部署",
        "check_test_service": '检测测试服务',
        "min_replicas": "最低副本数",
        "max_replicas": "最高副本数",
        "ports": "端口",
        "roll": "滚动发布",
        "k8s_yaml": "yaml",
        "service": "服务",
        "download_url": "下载地址",
        "metrics": "指标",
        "metrics_str": "指标",
        "md5": "md5",
        "service_type": "服务类型",
        "job_args_definition":"模板参数定义示例",
        "job_describe":"模板描述",
        "job_args_demo": "模板参数示例",
        "stop":"停止",
        "parallel_trial_count": "并行搜索次数",
        "max_trial_count": "最多搜索次数",
        "max_failed_trial_count": "最多失败搜索次数",
        "objective_type": "目标函数类型",
        "objective_goal": "目标值",
        "objective_metric_name": "目标度量",
        "objective_additional_metric_names": "附加目标度量",
        "algorithm_name": "搜索算法",
        "algorithm_setting": "搜索算法配置",
        "parameters": "超参数配置",
        "parameters_demo": "超参配置示例",
        "parameters_html": "超参数配置",
        "job_json": "搜索任务配置",
        "trial_spec": "任务 yaml",
        "trial_spec_html": "任务 yaml",
        "create_experiment": "启动调度",
        "run_instance":"运行实例",
        "monitoring": "监控",
        "monitoring_html":"监控",
        "link": "链接",
        "clear": "清理",
        "expand":"扩展",
        "expand_html":"扩展",
        "parameter": "扩展参数",
        "parameter_html": "扩展参数",
        "renew":"续期",
        "api_type":"接口类型",
        "code_dir":"代码目录",
        "id_url":"id",
        "debug": "调试",
        "run": "运行",
        "run_url":"运行",
        "depends_on_past":"过往依赖",
        "max_active_runs": "最大激活运行数",
        "des_image":"目标镜像",
        "target_image":"目标镜像",

        "creator": "创建者",
        "created_by": "创建者",
        "changed_by": "修改者",
        "created_on": "创建时间",
        "create_time": "创建时间",
        "changed_on": "修改时间",
        "change_time":"更新时间",
        "modified": "修改时间"
    }

    # 获取列的中文显示
    # @staticmethod
    def lab(col,label_columns=label_columns):
        if col in label_columns:
            return _(label_columns[col])
        return _(re.sub("[._]", " ", col).title())



    # 获取node选择器
    def get_default_node_selector(self,node_selector,resource_gpu,model_type):
        # 先使用项目中定义的选择器
        if not node_selector:
            node_selector=''

        # 不使用用户的填写，完全平台决定
        if core.get_gpu(resource_gpu)[0]:
            node_selector = node_selector.replace('cpu=true', 'gpu=true') + ",gpu=true,%s=true"%model_type
        else:
            node_selector = node_selector.replace('gpu=true', 'cpu=true') + ",cpu=true,%s=true"%model_type
        if 'org' not in node_selector:
            node_selector += ',org=public'
        node_selector = re.split(',|;|\n|\t', str(node_selector))
        node_selector = [selector.strip() for selector in node_selector if selector.strip()]
        node_selector = ','.join(list(set(node_selector)))
        return node_selector



