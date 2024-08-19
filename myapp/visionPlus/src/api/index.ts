import ajax from './ajax';

const getTemplateCommandConfig = (pipelineId: number | string, pipelineScenes: string): Promise<any> => {
  return ajax.get(`/${pipelineScenes}_modelview/api/template/list/${pipelineId}`);
};
// 保存配置
const pipeline_modelview_save = (pipelineId: number | string, pipelineScenes: string, data: Record<any, any>): Promise<any> => {
  return ajax.post({
    url: `/${pipelineScenes}_modelview/api/config/${pipelineId}`,
    data,
  });
};

// 获取流水线信息
const pipeline_modelview_detail = (pipelineId: number | string, pipelineScenes: string): Promise<any> => {
  return ajax.get(`/${pipelineScenes}_modelview/api/config/${pipelineId}`);
};



const api = {
  getTemplateCommandConfig,
  pipeline_modelview_save,
  pipeline_modelview_detail,
};

export default api;
