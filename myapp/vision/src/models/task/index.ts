import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import api from '@src/api';
import { RootState, AppThunk } from '../store';
import { selectPipelineId } from '../pipeline';

export interface TaskState {
  taskList: any;
  loading: boolean;
  taskId: number;
}
interface TaskList {
  id: string | number;
  changed: any;
}

const initialState: TaskState = {
  taskList: null,
  loading: false,
  taskId: 0,
};

const taskSlice = createSlice({
  name: 'task',
  initialState,
  reducers: {
    updateTaskList: (state, action: PayloadAction<TaskList>) => {
      if (!state.taskList) {
        const cur = new Map();
        cur.set(action.payload.id, action.payload.changed);
        state.taskList = cur;
      } else {
        state.taskList = state.taskList.set(action.payload.id, action.payload.changed);
      }
    },
    updateLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    resetTaskList: state => {
      state.taskList = new Map();
    },
    updateTaskId: (state, action: PayloadAction<number>) => {
      state.taskId = action.payload;
    },
  },
});

export const { updateTaskList, updateLoading, resetTaskList, updateTaskId } = taskSlice.actions;

export const selectTaskList = (state: RootState): TaskState['taskList'] => state.task.taskList;
export const selectLoading = (state: RootState): TaskState['loading'] => state.task.loading;
export const selectTaskId = (state: RootState): TaskState['taskId'] => state.task.taskId;

// 依次提交保存改动的 task 配置
export const saveTaskList = async (): Promise<AppThunk> => async (dispatch, getState) => {
  dispatch(updateLoading(true));
  const state = getState();
  const taskList = selectTaskList(state);
  const queue: any = [];
  // taskList 为 Map 结构
  taskList &&
    taskList.forEach((value: any, key: any) => {
      if (Object.keys(value).length > 0) {
        queue.push(api.task_modelview_edit(selectPipelineId(state), key, value));
      }
    });
  const result = await Promise.allSettled(queue);
  dispatch(updateLoading(false));
  dispatch(resetTaskList()); // 清空已更新的 changed
  return result;
};

export default taskSlice.reducer;
