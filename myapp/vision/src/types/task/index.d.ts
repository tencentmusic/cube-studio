// 新增 task 的参数类型
export interface ITaskAdd {
  job_template: number;
  label: string;
  name: string;
  node_selector: string;
  pipeline: number;
  resource_cpu: string;
  resource_memory: string;
  [propname: string]: any;
}

// task 可编辑参数类型
export interface ITaskEdit {
  args?: string;
  command?: string;
  label?: string;
  name?: string;
  node_selector?: string;
  resource_cpu?: string;
  resource_gpu?: string;
  resource_rdma?: string;
  resource_memory?: string;
  retry?: string;
  timeout?: string;
  volume_mount?: string;
  working_dir?: string;
}
