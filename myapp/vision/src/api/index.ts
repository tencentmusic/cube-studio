import ajax from './ajax';
import { IPipelineAdd, IPipelineEdit } from '../types/pipeline';
import { ITaskAdd, ITaskEdit } from '../types/task';

const QuestUrl = 'http://11.150.126.122:8081/api/'
const TestUrl = 'http://11.161.238.209:8080/api/'


// 注册场景 1
const featureRegisterScenePostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureRegisterScene`,
    data: {
      data: {
        ...data,
      },
    },
  });
}
// 注册模型 1
const featureRegisterModelConfigPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureRegisterModelConfig`,
    data: {
      data: {
        ...data,
      },
    },
  });
}


// 注册特征插件配置 1
const featureRegisterFeProcConfigQUEST = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureRegisterFeProcConfig`,
    data: {
      ...data,
    },
  });
}

// 更新特征配置
const featureUpdateIsomerismFeatureFeatureInfoQUEST = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureUpdateIsomerismFeatureFeatureInfo`,
    data: {
      data: {
        ...data,
      },
    },
  });
}


// 注册模型服务路由 1
const featureRegisterModelServRouterPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureRegisterModelServRouter`,
    data: {
      data: {
        ...data,
      },
    },
  });
}
// 更新模型服务路由 1
const featureUpdateModelConfigPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureUpdateModelConfig`,
    data: {
      data: {
        ...data,
      },
    },
  });
}

// 注册特征拉取服务路由 1
const featureRegisterFeatureServRouterPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureRegisterFeatureServRouter`,
    data: {
      data: {
        ...data,
      },
    },
  });
}

// 注册特征  
const featureRegisterIsomerismFeatureFeatureInfoPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureRegisterIsomerismFeatureFeatureInfo`,
    data: {
      data: {
        ...data,
      }
    }
  })
}

// 注册算子  1
const featureRegisterOpPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureRegisterOp`,
    data: {
      data: {
        ...data,
      }
    }
  })
}
// 更新特征集合配置  1
const featureUpdateSetPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureUpdateSet`,
    data: {
      data: {
        ...data,
      }
    }
  })
}
// 修改插件特征配置  1
const featureUpdateInputItemPost = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureUpdateInputItem`,
    data: {
      ...data,
    }
  })
}
// 修改算子特征配置  1
const featureUpdateOpItemPost = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureUpdateOpItem`,
    data: {
      ...data,
    }
  })
}



// 所有场景
const featureScenePagesDisplayPostQuest = (id: number, idTwo: number, rtx: any, business: number): Promise<any> => {
  return ajax.get(`${TestUrl}featureScenePagesDisplay?p=${id}&s=${idTwo}&rtx=${rtx}&business=${business}`);
}

// 获取场景  
const featureSceneDisplayGetQuest = (id: any): Promise<any> => {
  return ajax.get(`${TestUrl}featureSceneDisplay?id=${id}`);
};

// 输入特征配置  
const featureFeProcConfigDetailDisplayGetQuest = (id: any): Promise<any> => {
  return ajax.get(`${TestUrl}featureFeProcConfigDetailDisplay?feId=${id}`);
};

// 获取插件配置  
const featureFeProcConfigDisplayGetQuest = (id: any, server: string): Promise<any> => {
  return ajax.get(`${TestUrl}featureFeProcConfigDisplay?scene_id=${id}&table=${server}`);
};
// 获取特征配置  
const featureDisplaySetGetQuest = (id: any, server: any, threeBool: any): Promise<any> => {
  return ajax.get(`${TestUrl}featureDisplaySet?featureSetId=${id}&isomerism=${server}&isFeatureServRouter=${threeBool}`);
};
// 获取算子  
const featureDisplayOpGetQuest = (): Promise<any> => {
  return ajax.get(`${TestUrl}featureDisplayOp`);
};

// 生成配置  
const featureBuildConfigurationGetQuest = (id: any, versionDesc: string): Promise<any> => {
  return ajax.get(`${TestUrl}featureBuildConfiguration?sceneId=${id}&versionDesc=${versionDesc}`);
};






// 拉取app归属数据 1
const featureKVDataDisplayPostQuest = (data: any): Promise<any> => {
  return ajax.post({
    url: `${TestUrl}featureKVDataDisplay`,
    data: {
      ...data,
    },
  });
}




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
  return ajax.get('/job_template_modelview/api/', {
    params: {
      form_data: JSON.stringify({
        columns: ['project', 'name', 'version', 'describe', 'images', 'workdir', 'entrypoint', 'args', 'demo', 'env',
          'hostAliases', 'privileged', 'accounts', 'created_by', 'changed_by', 'created_on', 'changed_on',
          'expand'],
        str_related: 0
      })
    }
  });
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
  get_component_config2,

  featureRegisterScenePostQuest,
  featureRegisterModelConfigPostQuest,
  featureRegisterFeProcConfigQUEST,
  featureRegisterModelServRouterPostQuest,
  featureUpdateModelConfigPostQuest,
  featureRegisterFeatureServRouterPostQuest,
  featureKVDataDisplayPostQuest,
  featureRegisterOpPostQuest,
  featureUpdateSetPostQuest,
  featureUpdateInputItemPost,
  featureUpdateOpItemPost,
  featureRegisterIsomerismFeatureFeatureInfoPostQuest,
  featureSceneDisplayGetQuest,
  featureFeProcConfigDetailDisplayGetQuest,
  featureFeProcConfigDisplayGetQuest,
  featureDisplaySetGetQuest,
  featureDisplayOpGetQuest,
  featureScenePagesDisplayPostQuest,
  featureUpdateIsomerismFeatureFeatureInfoQUEST,
  featureBuildConfigurationGetQuest,

};

export default api;
