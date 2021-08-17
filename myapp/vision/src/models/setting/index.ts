import { createSlice } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface SettingState {
  show: boolean;
}

const initialState: SettingState = {
  show: false,
};

const settingSlice = createSlice({
  name: 'setting',
  initialState,
  reducers: {
    toggle: state => {
      state.show = !state.show;
    },
  },
});

export const { toggle } = settingSlice.actions;

export const selectShow = (state: RootState): SettingState['show'] => state.setting.show;

export default settingSlice.reducer;
