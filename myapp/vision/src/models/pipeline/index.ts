import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState, AppThunk } from '../store';
import { updateErrMsg } from '../app';
import { selectElements } from '../element';
import api from '@src/api';
import getParamter from '@src/utils/getParamter';

export interface PipelineState {
  pipelineId: number | string;
  info: any;
  saved: boolean;
  changed: any;
  editing: boolean;
  pipelineList: any[];
}

const initialState: PipelineState = {
  pipelineId: getParamter('pipeline_id', window.location.href, false)?.pipeline_id || '',
  info: {},
  saved: true,
  changed: {},
  editing: false,
  pipelineList: [],
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
    updateChanged: (state, action: PayloadAction<any>) => {
      state.changed = action.payload;
    },
    updateEditing: (state, action: PayloadAction<boolean>) => {
      state.editing = action.payload;
    },
    updatePipelineList: (state, action: PayloadAction<any[]>) => {
      state.pipelineList = action.payload;
    },
  },
});

export const { updatePipelineId, updateInfo, updateSaved, updateChanged, updateEditing, updatePipelineList } =
  pipelineSlice.actions;

export const selectPipelineId = (state: RootState): PipelineState['pipelineId'] => state.pipeline.pipelineId;
export const selectInfo = (state: RootState): PipelineState['info'] => state.pipeline.info;
export const selectSaved = (state: RootState): PipelineState['saved'] => state.pipeline.saved;
export const selectChanged = (state: RootState): PipelineState['changed'] => state.pipeline.changed;
export const selectEditing = (state: RootState): PipelineState['editing'] => state.pipeline.editing;
export const selectPipelineList = (state: RootState): PipelineState['pipelineList'] => state.pipeline.pipelineList;

export const getPipelineList = (): AppThunk => dispatch => {
  api.pipeline_modelview_list().then(res => {
    if (res?.status === 0) {
      const list = res.result.map((item: any) => {
        return {
          id: item.id,
          name: item.name,
          describe: item.describe,
          changed_on: item.changed_on,
        };
      });
      dispatch(updatePipelineList(list));
    }
  });
};

// 获取当前流水线信息
export const getPipeline = (): AppThunk => (dispatch, getState) => {
  const state = getState();
  const pipelineId = selectPipelineId(state);

  if (pipelineId) {
    api
      .pipeline_modelview_detail(pipelineId)
      .then((res: any) => {
        if (res?.status === 0) {
          dispatch(updateInfo(res?.result));
        }
      })
      .catch(err => {
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
  const pipelineId = selectPipelineId(state);

  if (!selectSaved(state) || !pipelineId) return;

  dispatch(updateSaved(false));
  api
    .pipeline_modelview_edit(pipelineId, {
      ...selectChanged(state),
      expand: JSON.stringify(selectElements(state)),
    })
    .then((res: any) => {
      if (res?.status === 0) {
        dispatch(updateInfo(res?.result));
        dispatch(updateChanged({}));
        dispatch(updateEditing(false));
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
};

export default pipelineSlice.reducer;
