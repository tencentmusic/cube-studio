import { configureStore, ThunkAction, Action } from '@reduxjs/toolkit';
import { enableMapSet } from 'immer';
import appReducer from './app';
import editorReducer from './editor';
import elementReducer from './element';
import pipelineReducer from './pipeline';
import settingReducer from './setting';
import taskReducer from './task';
import templateReduce from './template';

enableMapSet();

export const store = configureStore({
  reducer: {
    app: appReducer,
    editor: editorReducer,
    element: elementReducer,
    pipeline: pipelineReducer,
    setting: settingReducer,
    task: taskReducer,
    template: templateReduce,
  },
  middleware: getDefaultMiddleware =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});

export type AppDispatch = typeof store.dispatch;
export type RootState = ReturnType<typeof store.getState>;
export type AppThunk<ReturnType = void> = ThunkAction<ReturnType, RootState, unknown, Action<string>>;
