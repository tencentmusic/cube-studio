import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface TemplateState {
  show: boolean;
  callout: boolean;
  current: any;
  info: CurrentInfo;
}
export interface CurrentInfo {
  createdBy: string;
  describe: string;
  id: number;
  imagesName: string;
  lastChanged: string;
  name: string;
  version: string;
  [key: string]: any;
}

const initialState: TemplateState = {
  show: true,
  callout: true,
  current: null,
  info: {
    createdBy: '',
    describe: '',
    id: 69,
    imagesName: '',
    lastChanged: '',
    name: '',
    version: '',
  },
};

const templateSlice = createSlice({
  name: 'template',
  initialState,
  reducers: {
    toggle: state => {
      state.show = !state.show;
    },
    updateCallout: (state, action: PayloadAction<boolean>) => {
      state.callout = action.payload;
    },
    updateCurrent: (state, action: PayloadAction<any>) => {
      state.current = action.payload;
    },
    updateInfo: (state, action: PayloadAction<CurrentInfo>) => {
      state.info = action.payload;
    },
  },
});

export const { toggle, updateCallout, updateCurrent, updateInfo } = templateSlice.actions;

export const selectShow = (state: RootState): TemplateState['show'] => state.template.show;
export const selectCallout = (state: RootState): TemplateState['callout'] => state.template.callout;
export const selectCurrent = (state: RootState): TemplateState['current'] => state.template.current;
export const selectInfo = (state: RootState): TemplateState['info'] => state.template.info;

export default templateSlice.reducer;
