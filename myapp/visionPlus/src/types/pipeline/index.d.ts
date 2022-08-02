// 新建流水线参数类型
export interface IPipelineAdd {
  describe: string;
  name: string;
  node_selector: string;
  schedule_type: string;
  image_pull_policy: string;
  parallelism: number;
  project: number;
}

// 流水线可编辑参数类型
export interface IPipelineEdit {
  project?: string;
  name?: string;
  describe?: string;
  namespace?: string;
  schedule_type?: string;
  cron_time?: string;
  node_selector?: string;
  image_pull_policy?: string;
  parallelism?: number;
  dag_json?: string;
  global_env?: string;
  expand: string;
}
