import ajax from './ajax';
import { IPipelineAdd, IPipelineEdit } from '../types/pipeline';
import { ITaskAdd, ITaskEdit } from '../types/task';

const QuestUrl = 'http://11.150.126.122:8081/api/'
// 8.1 回滚策略
const rollback = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}rollback`,
    data: {
      "strategy_id": data,
      "opr": "rollback"
    }
  });
};
// 8.2 测试发布策略
const test_release = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}test_release`,
    data: {
      "strategy_id": data,
      "opr": "test_release"
    }
  });
};

// 8.3 正式发布策略
const real_release = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}real_release`,
    data: {
      "strategy_id": data,
      "opr": "real_release"
    }
  });
};

// 8.12 统一物料召回查询
const get_strategys = (data: any): Promise<any> => {
  // return ajax.get(`http://11.150.126.122:8081/api/get_strategys?opr=${"get_strategys"}&&strategy_ids=[1,2]`);
  return ajax.post({
    url: `${QuestUrl}get_strategys`,
    data: {
      ...data,
      "opr": "get_strategys"
    }
  });
};

const get_components_mark = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}get_components_mark`,
    data: {
      "components": data,
      "opr": "get_components_mark"
    }
  });
};

const get_component_config = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}get_component_config`,
    data: {
      "component_mark": data,
      "component": "recall",
      "opr": "get_component_config",
      "component_type": ""
    }
  });
};
// 注册辅助组件
const get_component_config2 = (data: any, data2: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}get_component_config`,
    data: {
      "component_mark": "",
      "component": data2,
      "opr": "get_component_config",
      "component_type": data
    }
  });
};
// 、、8.4 修改策略
const modify_strategy = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}modify_strategy`,
    data: {
      data: {
        ...data,
      },
      "opr": "modify_strategy"
    }
  });
};
// 8.5 添加策略
const add_strategy = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}add_strategy`,
    data: {
      data: {
        ...data,
      },
      "opr": "add_strategy"
    }
  });
};
// 8.7 注册召回组件
const register_recall_component = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}register_recall_component`,
    data: {
      ...data,
      "opr": "register_recall_component"
    }
  });
};
// 8.8 注册辅助组件
const register_assistant_component = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}register_assistant_component`,
    data: {
      ...data,
      "opr": "register_assistant_component"
    }
  });
};

const get_components_type = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}get_components_type`,
    data: {
      "components": [data],
      "opr": "get_components_type"
    }
  });
};
// 8.6 注册辅助组件类型
const register_assistant_component_type = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}register_assistant_component_type`,
    data: {
      ...data,
      "opr": "register_assistant_component_type"
    }
  });
};

// 8.9 配置校验
const config_check = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}config_check`,
    data: {
      ...data,
      "opr": "config_check"
    }
  });
};
// 8.10 组件信息展示
const get_components_info = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}get_components_info`,
    data: {
      ...data,
      "opr": "get_components_info"
    }
  });
};
// 8.13 修改组件信息
const mod_component_info = (data: any): Promise<any> => {
  return ajax.post({
    url: `${QuestUrl}mod_component_info`,
    data: {
      ...data,
      "opr": "mod_component_info"
    }
  });
};

// 获取任务模板列表
const job_template_modelview = (): Promise<any> => {
  return ajax.get('/job_template_modelview/api/');
};

const project_all = (): Promise<any> => {
  return ajax.get('/project_modelview/api/');
};
// 获取 org 项目组
const project_modelview = (): Promise<any> => {
  return ajax.get('/project_modelview/org/api/');
};

// 新增流水线
const pipeline_modelview_add = (data: IPipelineAdd): Promise<any> => {
  return ajax.post({ url: '/pipeline_modelview/api/', data });
};

// 获取流水线列表
const pipeline_modelview_demo = (): Promise<any> => {
  return ajax.get('/pipeline_modelview/demo/list/');
};

// 获取流水线列表
const pipeline_modelview_list = (): Promise<any> => {
  return ajax.get('/pipeline_modelview/my/list/');
};

const pipeline_modelview_all = (filters: string): Promise<any> => {
  return ajax.get(`/pipeline_modelview/api/?form_data=${filters}`);
};

// 获取流水线信息
const pipeline_modelview_detail = (pipelineId: number | string): Promise<any> => {
  return ajax.get(`/pipeline_modelview/api/${pipelineId}`);
};

// 删除指定流水线
const pipeline_modelview_delete = (pipelineId: number | string): Promise<any> => {
  return ajax.delete(`/pipeline_modelview/api/${pipelineId}`);
};

// 流水线编辑提交
const pipeline_modelview_edit = (pipelineId: number | string, data: IPipelineEdit): Promise<any> => {
  return ajax.put({
    url: `/pipeline_modelview/api/${pipelineId}`,
    data,
  });
};

// 运行流水线
const pipeline_modelview_run = (pipelineId: number | string): Promise<any> => {
  return ajax.post({
    url: `/pipeline_modelview/api/run_pipeline/${pipelineId}`,
  });
};

// 克隆流水线
const pipeline_modelview_copy = (pipelineId: number | string): Promise<any> => {
  return ajax.post({
    url: `/pipeline_modelview/api/copy_pipeline/${pipelineId}`,
  });
};

// 往流水线中添加task
const task_modelview_add = (pipelineId: number | string, data: ITaskAdd): Promise<any> => {
  return ajax.post({
    url: '/task_modelview/api/',
    data: {
      ...data,
      filters: [
        {
          col: 'pipeline',
          opr: 'rel_o_m',
          value: +pipelineId,
        },
      ],
    },
  });
};

// 获取流水线中相应的task
const task_modelview_get = (taskId: string | number): Promise<any> => {
  return ajax.get(`/task_modelview/api/${taskId}`);
};

// 删除对应的 task
const task_modelview_del = (taskId: string | number): Promise<any> => {
  return ajax.delete(`/task_modelview/api/${taskId}`);
};

// 编辑 task
const task_modelview_edit = (pipelineId: string | number, taskId: string | number, data: ITaskEdit): Promise<any> => {
  return ajax.put({
    url: `/task_modelview/api/${taskId}`,
    data: {
      ...data,
      filters: [
        {
          col: 'pipeline',
          opr: 'rel_o_m',
          value: +pipelineId,
        },
      ],
    },
  });
};

const api = {
  job_template_modelview,
  project_all,
  project_modelview,
  pipeline_modelview_add,
  pipeline_modelview_demo,
  pipeline_modelview_list,
  pipeline_modelview_all,
  pipeline_modelview_detail,
  pipeline_modelview_delete,
  pipeline_modelview_edit,
  pipeline_modelview_run,
  pipeline_modelview_copy,
  task_modelview_add,
  task_modelview_get,
  task_modelview_del,
  task_modelview_edit,
  get_strategys,
  get_components_mark,
  add_strategy,
  register_recall_component,
  register_assistant_component,
  get_components_type,
  register_assistant_component_type,
  get_components_info,
  modify_strategy,
  mod_component_info,

  config_check,
  rollback,
  test_release,
  real_release,
  get_component_config,
  get_component_config2
};

export default api;
