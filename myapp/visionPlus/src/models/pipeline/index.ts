import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState, AppThunk } from '../store';
import { updateErrMsg } from '../app';
import api from '@src/api';
import getParamter from '@src/utils/getParamter';

export interface PipelineState {
  pipelineId: number | string;
  scenes:string;
  info: any;
  saved: boolean;
  changed: any;
  editing: boolean;
  pipelineList: any[];
  all: any[] | undefined;
  isLoadError: boolean
}

const initialState: PipelineState = {
  pipelineId: getParamter('pipeline_id', window.location.href, false)?.pipeline_id || '',
  scenes: getParamter('scenes', window.location.href, false)?.scenes || '',
  info: {},
  saved: true,
  changed: {},
  editing: false,
  pipelineList: [],
  all: undefined,
  isLoadError: false
};

const pipelineSlice = createSlice({
  name: 'pipeline',
  initialState,
  reducers: {
    updatePipelineId: (state, action: PayloadAction<number | string>) => {
      state.pipelineId = action.payload;
    },
    updateInfo: (state, action: PayloadAction<any>) => {
      state.info = action.payload;
    },
    updateSaved: (state, action: PayloadAction<boolean>) => {
      state.saved = action.payload;
    },
    updateIsLoadError: (state, action: PayloadAction<boolean>) => {
      state.isLoadError = action.payload;
    },
    updateChanged: (state, action: PayloadAction<any>) => {
      state.changed = action.payload;
    },
    updateEditing: (state, action: PayloadAction<boolean>) => {
      state.editing = action.payload;
    },
    updatePipelineList: (state, action: PayloadAction<any[]>) => {
      state.pipelineList = action.payload;
    },
    updateAll: (state, action: PayloadAction<any[] | undefined>) => {
      state.all = action.payload;
    },
  },
});

export const {
  updatePipelineId,
  updateInfo,
  updateSaved,
  updateChanged,
  updateEditing,
  updatePipelineList,
  updateAll,
  updateIsLoadError
} = pipelineSlice.actions;

export const selectPipelineId = (state: RootState): PipelineState['pipelineId'] => state.pipeline.pipelineId;
export const selectPipelineScenes = (state: RootState): PipelineState['scenes'] => state.pipeline.scenes;
export const selectInfo = (state: RootState): PipelineState['info'] => state.pipeline.info;
export const selectSaved = (state: RootState): PipelineState['saved'] => state.pipeline.saved;
export const selectChanged = (state: RootState): PipelineState['changed'] => state.pipeline.changed;
export const selectEditing = (state: RootState): PipelineState['editing'] => state.pipeline.editing;
export const selectAll = (state: RootState): PipelineState['all'] => state.pipeline.all;
export const selectIsLoadError = (state: RootState): PipelineState['isLoadError'] => state.pipeline.isLoadError;


// 获取当前流水线信息
export const getPipeline = (): AppThunk => (dispatch, getState) => {
  const state = getState();
  const pipelineId = selectPipelineId(state);
  const pipelineScenes = selectPipelineScenes(state);

  if (pipelineId && pipelineScenes) {
    api
      .pipeline_modelview_detail(pipelineId,pipelineScenes)
      .then((res: any) => {
        if (res?.status === 0) {
          dispatch(updateInfo(res));
          // dispatch(updateInfo(res?.result));
        }
      })
      .catch(err => {
        dispatch(updateIsLoadError(true))
        if (err.response) {
          dispatch(
            updateErrMsg({
              msg: err?.response?.data?.message || '获取流水线信息失败',
            }),
          );
        }
      });
  }
};

// 保存流水线
export const savePipeline = (): AppThunk => (dispatch, getState) => {
  const state = getState();
  console.log('savePipeline', state)
  const initProject = selectInfo(state)
  const nodeMsg = state.element.elements.filter(item => item.type === 'dataSet')

  const argsMsg = state.task.taskList
  const upstreamMsg = JSON.parse(state.pipeline.changed.dag_json)
  const sourceConfig = state.pipeline.info.dag_json
  const pipelineChange = selectChanged(state)
  const nodeInit = {
    "location": [],
    "color": { color: '', bg: '' },
    "label": "",
    "template-group": "",
    "template": "",
    "task-config": {},
    "templte_common_ui_config": {},
    "templte_ui_config": {}
  }

  const tarRes: any = {}
  const tarArgs: any = {}
  argsMsg && argsMsg.forEach((value: any, key: any) => {
    const args = value
    tarArgs[key] = args
  });

  for (let i = 0; i < nodeMsg.length; i++) {
    const item: any = nodeMsg[i];
    const itemInfo = item.data.info
    const defaultArgs: any = {}

    Object.keys(item.data.config).forEach(key => {
      const tarObj = item.data.config[key]
      for (const k in tarObj) {
        if (Object.prototype.hasOwnProperty.call(tarObj, k)) {
          const config = tarObj[k];
          defaultArgs[k] = config.default
        }
      }
    })

    const currentArgs = Object.keys(tarArgs[item.id] || {}).length ? tarArgs[item.id] : ((sourceConfig[item.id] || {})['task-config'] || defaultArgs)

    tarRes[item.id] = {
      ...tarRes[item.id],
      label: item.data.label,
      location: [item.position.x, item.position.y],
      color: itemInfo.color,
      template: itemInfo.template,
      templte_common_ui_config: itemInfo.templte_common_ui_config,
      templte_ui_config: itemInfo.templte_ui_config,
      'template-group': itemInfo['template-group'],
      'task-config': currentArgs,
      upstream: []
    }
  }

  for (const key in upstreamMsg) {
    if (Object.prototype.hasOwnProperty.call(upstreamMsg, key)) {
      const value = upstreamMsg[key];
      tarRes[key] = {
        ...tarRes[key],
        ...value
      }
    }
  }

  const pipelineId = selectPipelineId(state);
  const pipelineScenes = selectPipelineScenes(state);

  if (!selectSaved(state) || !pipelineId) return;
  dispatch(updateSaved(false));

  const tarParam = {
    ...initProject,
    dag_json: tarRes,
    config: JSON.parse(pipelineChange.args || '{}')
  }

  console.log('tarRes', tarRes, tarParam)
  api
    .pipeline_modelview_save(pipelineId,pipelineScenes, tarParam)
    .then((res: any) => {
      dispatch(updateEditing(false));

      if (res?.status === 0) {
        // dispatch(updateInfo(res?.result));
        // dispatch(updateChanged({}));
        // dispatch(updateEditing(false));
      }
    })
    .catch(err => {
      if (err.response) {
        dispatch(updateErrMsg({ msg: err.response.data.message }));
      }
    })
    .finally(() => {
      dispatch(updateSaved(true));
    });
  dispatch(updateSaved(true));
};

export default pipelineSlice.reducer;
